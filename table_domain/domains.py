import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage

G1_ENUM = frozenset([
    "defi", "nft", "lending", "derivatives", "bridge",
    "staking", "oracle", "governance", "identity", "gaming",
    "social", "infra", "payments", "rwa", "analytics",
])

G2_CAP = 225
G3_CAP = 1275
DEFAULT_BATCH = 25  # intentionally small — large batches cause LLM to drop or misorient rows

DEFAULT_INPUT    = "processed_sources.csv"
DEFAULT_OUTPUT   = "table_domain/taxonomy_output.csv"
DEFAULT_REGISTRY = "table_domain/taxonomy_registry.json"

SYSTEM_PROMPT = """\
You are a **Blockchain Data Ontology Compiler**.

Map each input (namespace, table_name) into a 3-layer taxonomy:
  G1 = Sector          — FIXED ENUM ONLY (15 values)
  G2 = System Category — GENERATED, global unique cap = 225
  G3 = Functional Primitive — atomic verb-first action, global cap = 1275

## G1 — FIXED ENUM (pick exactly one)
defi | nft | lending | derivatives | bridge | staking | oracle | governance | identity | gaming | social | infra | payments | rwa | analytics

## G2 RULES
- System-level cluster inside a sector
- MUST be a noun phrase describing a system type, NOT a verb/action
- Snake_case, lowercase
- Reuse an existing G2 ONLY if it is a precise, specific match for this exact system type — do NOT stretch existing G2s to cover loosely related systems
- Create a NEW G2 whenever the existing registry does not have a precise fit; target ~1 G2 per 40–50 rows processed
- INVALID (verb territory): swap, mint, borrow, transfer
- VALID examples: dex_amm, dex_orderbook, dex_aggregator, liquidity_pools,
  liquidity_provisioning, yield_vaults, structured_products, money_markets,
  collateralized_lending, liquidation_systems, credit_delegation,
  nft_marketplaces, nft_minting_systems, nft_rentals, nftfi_protocols,
  proposal_systems, voting_systems, execution_frameworks, bridge_protocols,
  staking_pools, oracle_feeds, identity_registries, gaming_assets,
  social_graphs, token_contracts, fee_systems, reward_systems

## G3 RULES
- Atomic event-level action (state transition)
- verb-first, lowercase + underscores
- Single action only — no compound events
- MUST NOT overlap with G2 meaning
- Reuse an existing G3 ONLY if it is a precise, specific match — do NOT stretch to cover loosely related actions
- Create a NEW G3 whenever the existing registry lacks a precise fit; target ~1 G3 per 7 rows processed
- Examples: asset_exchange, liquidity_added, liquidity_removed,
  debt_origination, debt_repayment, collateral_seizure,
  nft_mint, nft_transfer, royalty_distribution,
  vote_cast, proposal_creation, proposal_execution,
  token_deposit, token_withdrawal, reward_claimed,
  bridge_initiated, bridge_completed, stake_deposited,
  stake_withdrawn, price_updated, address_labeled

## OUTPUT
Return ONLY a JSON array — one object per input row, in the EXACT same order as the input.
The array MUST have the same number of elements as the input.
Each object: {{"namespace": "...", "table_name": "...", "g1": "...", "g2": "...", "g3": "..."}}
No markdown, no explanation, no preamble.
"""

HUMAN_PROMPT = """\
## EXISTING REGISTRY (reuse ONLY on precise match — do NOT stretch to cover loosely related systems/actions)
{registry_json}

## DIVERSITY BUDGET
G2 used: {g2_used}/{g2_cap} — remaining: {g2_remaining}
G3 used: {g3_used}/{g3_cap} — remaining: {g3_remaining}
You MUST introduce at least {min_new_g2} new G2(s) and {min_new_g3} new G3(s) in this batch unless the cap is reached.
If a system or action is even slightly distinct from existing entries, create a new label.

## INPUT ({count} rows — return exactly {count} results)
{batch_json}
"""

class TaxonomyRegistry:
    def __init__(self, path: str):
        self.path = Path(path)
        self.g2_set: set[str] = set()
        self.g3_set: set[str] = set()

    def load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.g2_set = set(data.get("g2", []))
            self.g3_set = set(data.get("g3", []))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "g2": sorted(self.g2_set),
            "g3": sorted(self.g3_set),
        }, indent=2))

    def to_prompt_json(self) -> str:
        return json.dumps({"g2": sorted(self.g2_set), "g3": sorted(self.g3_set)}, indent=2)

    @property
    def g2_count(self) -> int:
        return len(self.g2_set)

    @property
    def g3_count(self) -> int:
        return len(self.g3_set)

    def register(self, g2: str, g3: str) -> None:
        self.g2_set.add(g2)
        self.g3_set.add(g3)



def make_llm() -> ChatOpenRouter:
    return ChatOpenRouter(
        model="moonshotai/kimi-k2",
        temperature=0,
        openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
    )

def classify_batch(
    llm: ChatOpenRouter,
    registry: TaxonomyRegistry,
    rows: list[dict],
    total_remaining: int,
) -> list[dict]:
    """Call the LLM for a batch of rows. Returns classified rows."""
    batch_items = [{"namespace": r["namespace"], "table_name": r["table_name"]} for r in rows]
    batch_json = json.dumps(batch_items, indent=2)
    g2_remaining = max(0, G2_CAP - registry.g2_count)
    g3_remaining = max(0, G3_CAP - registry.g3_count)
    # proportional targets: spread the remaining budget evenly over remaining rows
    batch_fraction = len(rows) / max(total_remaining, len(rows))
    min_new_g2 = min(max(0, int(g2_remaining * batch_fraction)), g2_remaining)
    min_new_g3 = min(max(1, round(g3_remaining * batch_fraction)), g3_remaining)
    human_text = HUMAN_PROMPT.format(
        registry_json=registry.to_prompt_json(),
        batch_json=batch_json,
        count=len(rows),
        g2_used=registry.g2_count,
        g2_cap=G2_CAP,
        g2_remaining=g2_remaining,
        g3_used=registry.g3_count,
        g3_cap=G3_CAP,
        g3_remaining=g3_remaining,
        min_new_g2=min_new_g2,
        min_new_g3=min_new_g3,
    )
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_text),
    ])
    raw = response.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    classified = json.loads(raw)

    # If LLM returned wrong number of items, do positional alignment
    if len(classified) != len(rows):
        print(
            f"  Warning: LLM returned {len(classified)} items for {len(rows)} input rows — aligning by position",
            file=sys.stderr,
        )
        aligned = []
        for i, orig in enumerate(batch_items):
            if i < len(classified):
                item = classified[i]
                # Overwrite namespace/table_name with ground truth from input
                item["namespace"] = orig["namespace"]
                item["table_name"] = orig["table_name"]
                aligned.append(item)
            else:
                aligned.append({**orig, "g1": "infra", "g2": "token_contracts", "g3": "state_transition"})
        classified = aligned

    return classified


def validate_and_fix(item: dict, registry: TaxonomyRegistry) -> dict:
    """Validate G1/G2/G3 and fix obvious issues."""
    g1 = item.get("g1", "infra").lower().strip()
    g2 = item.get("g2", "token_contracts").lower().strip()
    g3 = item.get("g3", "state_transition").lower().strip()

    # G1 must be in the fixed enum
    if g1 not in G1_ENUM:
        g1 = "infra"

    # G2 cap enforcement: if at cap and g2 is new, pick closest existing by prefix then g1 sector
    if g2 not in registry.g2_set and registry.g2_count >= G2_CAP:
        prefix = g2.split("_")[0]
        candidates = sorted(x for x in registry.g2_set if x.startswith(prefix))
        if not candidates:
            # Fall back to entries that share the g1 sector name as a prefix
            candidates = sorted(x for x in registry.g2_set if x.startswith(g1))
        g2 = candidates[0] if candidates else sorted(registry.g2_set)[0]

    # G3 cap enforcement
    if g3 not in registry.g3_set and registry.g3_count >= G3_CAP:
        prefix = g3.split("_")[0]
        candidates = sorted(x for x in registry.g3_set if x.startswith(prefix))
        g3 = candidates[0] if candidates else sorted(registry.g3_set)[0]

    registry.register(g2, g3)

    return {
        "namespace":  item.get("namespace", ""),
        "table_name": item.get("table_name", ""),
        "g1": g1,
        "g2": g2,
        "g3": g3,
    }


def run(
    input_path: str,
    output_path: str,
    registry_path: str,
    batch_size: int,
    resume: bool,
) -> None:
    registry = TaxonomyRegistry(registry_path)
    registry.load()
    llm = make_llm()
    # Read input CSV
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    print(f"Total rows: {len(all_rows)}", file=sys.stderr)

    # Determine already-processed rows if resuming
    done_keys: set[tuple[str, str]] = set()
    if resume and Path(output_path).exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_keys.add((row["namespace"], row["table_name"]))
        print(f"Resuming — skipping {len(done_keys)} already processed rows", file=sys.stderr)

    pending = [
        r for r in all_rows
        if (r["namespace"], r["table_name"]) not in done_keys
    ]
    print(f"Rows to classify: {len(pending)}", file=sys.stderr)

    # Open output file (append if resuming, else overwrite)
    mode = "a" if resume and Path(output_path).exists() else "w"
    out_file = open(output_path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=["namespace", "table_name", "g1", "g2", "g3"])
    if mode == "w":
        writer.writeheader()

    try:
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            batch_end = batch_start + len(batch)
            print(
                f"Classifying rows {batch_start + 1}–{batch_end} / {len(pending)} "
                f"| G2 unique: {registry.g2_count}/{G2_CAP} "
                f"| G3 unique: {registry.g3_count}/{G3_CAP}",
                file=sys.stderr,
            )

            rows_remaining = len(pending) - batch_start
            for attempt in range(3):
                try:
                    classified = classify_batch(llm, registry, batch, rows_remaining)
                    break
                except Exception as e:
                    print(f"  Attempt {attempt + 1} failed: {e}", file=sys.stderr)
                    if attempt == 2:
                        classified = [
                            {**r, "g1": "infra", "g2": "token_contracts", "g3": "state_transition"}
                            for r in batch
                        ]
                    else:
                        time.sleep(2 ** attempt)

            key_map = {(r["namespace"], r["table_name"]): r for r in batch}
            for item in classified:
                ns  = item.get("namespace", "")
                tbl = item.get("table_name", "")
                # ensure we have correct namespace/table_name from original input
                original = key_map.get((ns, tbl))
                if original:
                    item["namespace"]  = original["namespace"]
                    item["table_name"] = original["table_name"]

                validated = validate_and_fix(item, registry)
                writer.writerow(validated)

            out_file.flush()
            registry.save()

    finally:
        out_file.close()

    print(
        f"\nDone. Output: {output_path}\n"
        f"G2 unique: {registry.g2_count}/{G2_CAP}\n"
        f"G3 unique: {registry.g3_count}/{G3_CAP}",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify blockchain source tables into G1/G2/G3 taxonomy via LLM."
    )
    parser.add_argument("--input",      default=DEFAULT_INPUT,    help="Input CSV (namespace,table_name)")
    parser.add_argument("--output",     default=DEFAULT_OUTPUT,   help="Output CSV")
    parser.add_argument("--registry",   default=DEFAULT_REGISTRY, help="Registry JSON (persists G2/G3 sets)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Rows per LLM call")
    parser.add_argument("--resume",     action="store_true", help="Skip already-output rows")
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        registry_path=args.registry,
        batch_size=args.batch_size,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
