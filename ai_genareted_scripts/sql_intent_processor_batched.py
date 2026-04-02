"""
SQL Intent Processor with Ollama (Optimized + Batch Prompting)
===============================================================
Processes parquet files using Ollama for natural language generation.
Sends multiple SQLs per request for 3-5x speedup.
Uses same system prompt as sql_narrator_ollama.
"""

import pandas as pd
import os
import json
import re
import httpx
from io import StringIO
from tqdm import tqdm
from test_scripts.sql_narrator_ollama import SQLNarrator, OllamaConfig, SYSTEM_PROMPT


# ========================
# Configuration
# ========================
CONFIG = {
    # Ollama settings
    "ollama_model": "llama3-gradient",
    "ollama_base_url": "http://192.168.1.76:8080",
    
    # Processing settings
    "queries_per_batch": 7,
    "batch_write_size": 7,
    "checkpoint_interval": 7,
    "max_sql_length": 80000,
    
    # Paths
    "parquet_path": "dataset/5839995-5849994.parquet",
    "output_file": "query_intents_output.txt",
    "checkpoint_file": "checkpoint.json",
}


# ========================
# Batch Prompt Template (uses imported SYSTEM_PROMPT)
# ========================

BATCH_USER_PROMPT = """
Convert SQL queries to pseudo-function pipelines.

CTE = function call, chain with →, use $var for parameters.
Focus on INTENT, not literal translation. Create new functions if needed.

Variables:
$token_address      - ERC20 token contract
$wallet_address     - user wallet
$contract_address   - smart contract
$collection_address - NFT collection
$pool_address       - LP pool

$chain              - blockchain (ethereum, polygon, arbitrum, optimism, base, bsc, all)
$source_chain       - bridge origin
$dest_chain         - bridge destination

$from_date          - start date
$to_date            - end date

$dex_name           - DEX (uniswap, sushiswap, curve)
$protocol_name      - protocol (aave, compound, lido)
$bridge_name        - bridge (stargate, hop, across)
$marketplace        - NFT market (opensea, blur)
$stablecoin_symbol  - stablecoin (USDT, USDC, DAI)

$limit              - result limit
$min_balance        - minimum balance filter
$threshold          - numeric threshold

Format: ID|category|chain|function_pipeline|variables

Examples:
123|token|ethereum|raw_holders(token=$token_address) → rank(by=balance, limit=$limit) → leaderboard()|token_address,limit
456|dex|arbitrum|raw_swaps(dex=uniswap, token=$token_address, from=$from_date, to=$to_date) → group_by(day) → aggregate(sum, volume) → timeseries()|token_address,from_date,to_date
789|stablecoin|all|raw_stablecoin_flows(from=$from_date, to=$to_date) → group_by(symbol) → aggregate(sum, volume) → market_share()|from_date,to_date
321|bridge|all|raw_bridge_transfers(bridge=$bridge_name, chain_from=$source_chain, chain_to=$dest_chain) → aggregate(sum, volume) → total()|bridge_name,source_chain,dest_chain
555|lending|polygon|raw_borrows(protocol=aave, from=$from_date, to=$to_date) → group_by(token) → aggregate(sum, amount) → tvl()|from_date,to_date
666|nft|ethereum|raw_nft_sales(collection=$collection_address, from=$from_date, to=$to_date) → group_by(day) → aggregate(sum, price_eth) → timeseries()|collection_address,from_date,to_date

{sql_list}

{output_format}
"""

# ========================
# Checkpoint Management
# ========================
def load_checkpoint(checkpoint_file: str) -> set:
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(checkpoint_file: str, processed_ids: set):
    temp_file = checkpoint_file + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(list(processed_ids), f)
    os.replace(temp_file, checkpoint_file)


# ========================
# Batch Iterator
# ========================
def batch_iterator(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# ========================
# Batch Ollama Call (with system prompt)
# ========================
def call_ollama_batch(
    sqls: list[tuple[str, str]],
    base_url: str,
    model: str,
    max_sql_len: int,
    system_prompt: str
) -> dict[str, str]:
    """
    Send multiple SQLs in one request with system prompt.
    Returns: {query_id: explanation}
    """
    sql_list = ""
    output_format = ""
    id_map = {}
    
    for i, (query_id, sql) in enumerate(sqls, 1):
        truncated = sql[:max_sql_len] if len(sql) > max_sql_len else sql
        sql_list += f"Query {query_id}:\n```sql\n{truncated}\n```\n\n"
        output_format += f"{query_id}: <explanation>\n"
        id_map[i] = query_id
    
    user_prompt = BATCH_USER_PROMPT.format(sql_list=sql_list, output_format=output_format)
    
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 80 * len(sqls),
                        "num_ctx": 10000,
                        "temperature": 0.1,
                        "num_thread": 16,
                        "num_gpu": 99,
                    }
                }
            )
            response.raise_for_status()
            text = response.json().get("response", "")
            print(f"Ollama response:\n{text}")
    except Exception as e:
        return {qid: f"Error: {str(e)}" for qid in id_map.values()}
    
    # Parse output using query IDs
    results = {}
    for i, query_id in id_map.items():
        # Match "query_id: explanation"
        pattern = rf"^{re.escape(str(query_id))}[:.]\s*(.+)$"
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            results[query_id] = match.group(1).strip()
        else:
            results[query_id] = f"Parse error (query {query_id})"
    
    return results


# ========================
# Batched Pipeline (FAST)
# ========================
def process_queries_batched(
    parquet_path: str,
    output_file: str,
    checkpoint_file: str,
    ollama_model: str,
    ollama_base_url: str,
    queries_per_batch: int,
    batch_write_size: int,
    checkpoint_interval: int,
    max_sql_length: int,
):
    """
    Process SQL queries in batches (3-5x faster).
    Uses same system prompt as SQLNarrator.
    """
    print("Loading data...")
    df = pd.read_parquet(parquet_path, columns=['query_id', 'query_sql'])
    processed_ids = load_checkpoint(checkpoint_file)
    
    remaining = df[~df['query_id'].isin(processed_ids)][['query_id', 'query_sql']].values.tolist()
    total_queries = len(remaining)
    
    print(f"Total: {len(df)} | Done: {len(processed_ids)} | Remaining: {total_queries}")
    print(f"Using system prompt from sql_narrator_ollama.py")
    print(f"Model: {ollama_model} | Batch size: {queries_per_batch}")
    
    if not remaining:
        print("All done!")
        return
    
    num_batches = (total_queries + queries_per_batch - 1) // queries_per_batch
    
    buffer = StringIO()
    buffer_count = 0
    total_processed = 0
    
    with open(output_file, "a", encoding="utf-8") as f:
        with tqdm(total=num_batches, desc="Batches", unit="batch") as pbar:
            for batch in batch_iterator(remaining, queries_per_batch):
                
                sqls = [(str(qid), sql) for qid, sql in batch]
                results = call_ollama_batch(
                    sqls,
                    ollama_base_url,
                    ollama_model,
                    max_sql_length,
                    SYSTEM_PROMPT  # <-- Uses imported system prompt
                )
                
                for query_id, explanation in results.items():
                    output = (
                        f"--- Query ID: {query_id} ---\n"
                        f"Query: {explanation}\n"
                        f"{'-' * 50}\n\n"
                    )
                    buffer.write(output)
                    buffer_count += 1
                    processed_ids.add(query_id)
                    total_processed += 1
                
                if buffer_count >= batch_write_size:
                    f.write(buffer.getvalue())
                    f.flush()
                    buffer = StringIO()
                    buffer_count = 0
                
                if total_processed % checkpoint_interval == 0:
                    save_checkpoint(checkpoint_file, processed_ids)
                
                pbar.update(1)
        
        if buffer_count > 0:
            f.write(buffer.getvalue())
            f.flush()
    
    save_checkpoint(checkpoint_file, processed_ids)
    print(f"\nDone! Processed {total_processed} queries in {num_batches} batches.")


# ========================
# Sequential Pipeline (Original)
# ========================
def process_queries_sequential(
    parquet_path: str,
    output_file: str,
    checkpoint_file: str,
    ollama_model: str,
    batch_write_size: int,
    checkpoint_interval: int,
):
    """
    Process SQL queries sequentially using SQLNarrator.
    """
    config = OllamaConfig(model=ollama_model)
    narrator = SQLNarrator(config)
    
    print("Loading data...")
    df = pd.read_parquet(parquet_path, columns=['query_id', 'query_sql'])
    processed_ids = load_checkpoint(checkpoint_file)
    
    remaining = df[~df['query_id'].isin(processed_ids)]
    print(f"Total: {len(df)} | Processed: {len(processed_ids)} | Remaining: {len(remaining)}")
    
    if len(remaining) == 0:
        print("All done!")
        return
    
    narrator._check_ollama()
    
    buffer = StringIO()
    buffer_count = 0
    total_processed = 0
    
    with open(output_file, "a", encoding="utf-8") as f:
        with tqdm(total=len(remaining), desc="Processing", unit="query") as pbar:
            for _, row in remaining.iterrows():
                query_id = row['query_id']
                sql_query = row['query_sql']
                
                try:
                    narrative = narrator.narrate(sql_query)
                except Exception as e:
                    narrative = f"Error: {str(e)}"
                
                output_block = (
                    f"--- Query ID: {query_id} ---\n"
                    f"{narrative}\n"
                    f"{'-' * 50}\n\n"
                )
                
                buffer.write(output_block)
                buffer_count += 1
                processed_ids.add(query_id)
                total_processed += 1
                
                if buffer_count >= batch_write_size:
                    f.write(buffer.getvalue())
                    f.flush()
                    buffer = StringIO()
                    buffer_count = 0
                
                if total_processed % checkpoint_interval == 0:
                    save_checkpoint(checkpoint_file, processed_ids)
                
                pbar.update(1)
        
        if buffer_count > 0:
            f.write(buffer.getvalue())
            f.flush()
    
    save_checkpoint(checkpoint_file, processed_ids)
    print(f"\nDone! Processed {len(remaining)} queries.")


# ========================
# Entry Point
# ========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process SQL queries to natural language")
    parser.add_argument("--mode", choices=["batched", "sequential"], default="batched",
                        help="Processing mode: batched (fast) or sequential")
    parser.add_argument("--model", default=CONFIG["ollama_model"], help="Ollama model name")
    parser.add_argument("--queries-per-batch", type=int, default=CONFIG["queries_per_batch"], help="SQLs per Ollama request")
    parser.add_argument("--batch-write", type=int, default=CONFIG["batch_write_size"], help="Batch write size")
    parser.add_argument("--checkpoint-interval", type=int, default=CONFIG["checkpoint_interval"])
    parser.add_argument("--input", default=CONFIG["parquet_path"], help="Input parquet file")
    parser.add_argument("--output", default=CONFIG["output_file"], help="Output text file")
    
    args = parser.parse_args()
    
    if args.mode == "batched":
        process_queries_batched(
            parquet_path=args.input,
            output_file=args.output,
            checkpoint_file=CONFIG["checkpoint_file"],
            ollama_model=args.model,
            ollama_base_url=CONFIG["ollama_base_url"],
            queries_per_batch=args.queries_per_batch,
            batch_write_size=args.batch_write,
            checkpoint_interval=args.checkpoint_interval,
            max_sql_length=CONFIG["max_sql_length"],
        )
    else:
        process_queries_sequential(
            parquet_path=args.input,
            output_file=args.output,
            checkpoint_file=CONFIG["checkpoint_file"],
            ollama_model=args.model,
            batch_write_size=args.batch_write,
            checkpoint_interval=args.checkpoint_interval,
        )