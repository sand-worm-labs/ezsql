"""
SQL Query Classifier
====================
Classifies Dune SQL queries into:
  - chains touched (e.g. ethereum,arbitrum)
  - canonical analytics pattern (e.g. wallet profiling, dex trade history)

Output format (TSV):
  query_id	chains	label
  12345		ethereum,arbitrum	fund flow tracing
"""

import pandas as pd
import os
import json
import re
import httpx
from io import StringIO
from tqdm import tqdm


# ========================
# Configuration
# ========================
CONFIG = {
    "ollama_model": "llama3-gradient",
    "ollama_base_url": "http://192.168.1.76:8080",
    "queries_per_batch": 5,
    "batch_write_size": 5,
    "checkpoint_interval": 5,
    "max_sql_length": 8000,
    "parquet_path": "dataset/5839995-5849994.parquet",
    "output_file": "query_classifications.tsv",
    "checkpoint_file": "classifier_checkpoint.json",
}


# ========================
# Prompts
# ========================
SYSTEM_PROMPT = """You are a blockchain analytics expert. You classify SQL queries from Dune Analytics.
Your job: identify which chains the query touches and what analytical pattern it represents.
Be concise. One line per query. No explanation."""

CHAINS = [
    "ethereum", "arbitrum", "optimism", "polygon", "base", "bsc",
    "avalanche", "solana", "fantom", "gnosis", "zksync", "linea", "scroll"
]

LABELS = [
    "wallet profiling",
    "fund flow tracing",
    "token holder distribution",
    "dex trade history",
    "contract event parsing",
    "liquidity pool snapshot",
    "bridge volume",
    "nft sales",
    "lending/borrowing",
    "stablecoin flow",
    "protocol revenue",
    "governance activity",
    "token price/market",
]

BATCH_PROMPT_TEMPLATE = """Classify each SQL query below.

For each query output EXACTLY one line:
  <query_id> | <comma-separated chains> | <label>

Chains (only list chains actually referenced in the query):
{chains}

Labels (pick the single best match):
{labels}

Examples:
42 | ethereum | wallet profiling
77 | ethereum,arbitrum | fund flow tracing
99 | ethereum,polygon,optimism | bridge volume
11 | solana | nft sales

Queries:
{{sql_list}}
""".format(
    chains=", ".join(CHAINS),
    labels=", ".join(LABELS),
)


# ========================
# Checkpoint
# ========================
def load_checkpoint(path: str) -> set:
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(path: str, done: set):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(list(done), f)
    os.replace(tmp, path)


# ========================
# Batch iterator
# ========================
def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ========================
# Ollama call
# ========================
def classify_batch(
    sqls: list[tuple[str, str]],
    base_url: str,
    model: str,
    max_sql_len: int,
) -> dict[str, tuple[str, str]]:
    sql_list = ""
    for query_id, sql in sqls:
        truncated = sql[:max_sql_len]
        sql_list += f"[{query_id}]\n```sql\n{truncated}\n```\n\n"

    prompt = BATCH_PROMPT_TEMPLATE.format(sql_list=sql_list)

    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "system": SYSTEM_PROMPT,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 60 * len(sqls),
                        "num_ctx": 12000,
                        "temperature": 0.0,
                        "num_thread": 16,
                        "num_gpu": 99,
                    }
                }
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")
    except Exception as e:
        print(f"Ollama error: {e}")
        return {qid: ("error", "error") for qid, _ in sqls}

    results = {}
    valid_chains = set(CHAINS)

    for query_id, _ in sqls:
        pattern = rf"^{re.escape(str(query_id))}\s*\|\s*([^\|]+)\s*\|\s*(.+)$"
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            raw_chains = match.group(1).strip().lower()
            raw_label = match.group(2).strip().lower()

            # Filter to known chains only, but keep raw if none match
            found_chains = [c.strip() for c in raw_chains.split(",") if c.strip() in valid_chains]
            chains = ",".join(found_chains) if found_chains else raw_chains

            # Trust the label as-is — no unknown fallback
            results[query_id] = (chains, raw_label)
        else:
            results[query_id] = ("parse error", "parse error")

    return results


# ========================
# Main pipeline
# ========================
def classify_queries(
    parquet_path: str,
    output_file: str,
    checkpoint_file: str,
    model: str,
    base_url: str,
    queries_per_batch: int,
    batch_write_size: int,
    checkpoint_interval: int,
    max_sql_length: int,
):
    print("Loading data...")
    df = pd.read_parquet(parquet_path, columns=["query_id", "query_sql"])
    done = load_checkpoint(checkpoint_file)

    remaining = df[~df["query_id"].isin(done)][["query_id", "query_sql"]].values.tolist()
    total = len(remaining)

    print(f"Total: {len(df)} | Done: {len(done)} | Remaining: {total}")
    print(f"Model: {model} | Batch size: {queries_per_batch}")

    if not remaining:
        print("All done!")
        return

    write_header = not os.path.exists(output_file)
    buffer = StringIO()
    buf_count = 0
    processed = 0

    with open(output_file, "a", encoding="utf-8") as f:
        if write_header:
            f.write("query_id\tchains\tlabel\n")

        num_batches = (total + queries_per_batch - 1) // queries_per_batch

        with tqdm(total=num_batches, desc="Classifying", unit="batch") as pbar:
            for batch in batches(remaining, queries_per_batch):
                sqls = [(str(qid), sql) for qid, sql in batch]
                results = classify_batch(sqls, base_url, model, max_sql_length)

                for query_id, (chains, label) in results.items():
                    buffer.write(f"{query_id}\t{chains}\t{label}\n")
                    buf_count += 1
                    done.add(query_id)
                    processed += 1

                if buf_count >= batch_write_size:
                    f.write(buffer.getvalue())
                    f.flush()
                    buffer = StringIO()
                    buf_count = 0

                if processed % checkpoint_interval == 0:
                    save_checkpoint(checkpoint_file, done)

                pbar.update(1)

        if buf_count > 0:
            f.write(buffer.getvalue())
            f.flush()

    save_checkpoint(checkpoint_file, done)
    print(f"\nDone! Classified {processed} queries → {output_file}")


# ========================
# Entry point
# ========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify Dune SQL queries")
    parser.add_argument("--model", default=CONFIG["ollama_model"])
    parser.add_argument("--input", default=CONFIG["parquet_path"])
    parser.add_argument("--output", default=CONFIG["output_file"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["queries_per_batch"])
    args = parser.parse_args()

    classify_queries(
        parquet_path=args.input,
        output_file=args.output,
        checkpoint_file=CONFIG["checkpoint_file"],
        model=args.model,
        base_url=CONFIG["ollama_base_url"],
        queries_per_batch=args.batch_size,
        batch_write_size=CONFIG["batch_write_size"],
        checkpoint_interval=CONFIG["checkpoint_interval"],
        max_sql_length=CONFIG["max_sql_length"],
    )