import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage

# ─── CONSTANTS ───────────────────────────────────────────────────────────────

G1_ENUM = frozenset([
    "defi", "nft", "lending", "derivatives", "bridge",
    "staking", "oracle", "governance", "identity", "gaming",
    "social", "infra", "payments", "rwa", "analytics",
])

G2_CAP = 225
G3_CAP = 1275

DEFAULT_INPUT    = "processed_sources.csv"
DEFAULT_OUTPUT   = "table_domain/taxonomy_output.csv"
DEFAULT_REGISTRY = "table_domain/taxonomy_registry.json"
DEFAULT_BATCH    = 25

FALLBACK_G1 = "infra_fallback"
FALLBACK_G2 = "token_contracts_fallback"
FALLBACK_G3 = "token_transferred_fallback"

STALL_THRESHOLD = 3


# ─── PROMPTS ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a **Blockchain Data Ontology Compiler**.

Map each input (namespace, table_name) into a 3-layer taxonomy:
  G1 = Sector              — FIXED ENUM ONLY (15 values)
  G2 = System Category     — noun phrase describing a system type, global cap = {g2_cap}
  G3 = Functional Primitive — atomic verb-first action, global cap = {g3_cap}

────────────────────────────────────────────
## G1 — FIXED ENUM (pick exactly one)
defi | nft | lending | derivatives | bridge | staking | oracle
governance | identity | gaming | social | infra | payments | rwa | analytics

────────────────────────────────────────────
## G2 RULES
- System-level cluster inside a sector
- MUST be a noun phrase describing a system type — NOT a verb, action, or protocol name
- snake_case, lowercase
- Reuse an existing G2 ONLY on a precise, specific match — do NOT stretch to cover loosely related systems
- Create a NEW G2 whenever the existing registry lacks a precise fit
- Target: ~1 new G2 per 40–50 rows processed
- INVALID: protocol-specific names (e.g. benqi_yield_strategies, aave_yield_strategies)
- INVALID: verb-first labels (swap, mint, borrow, transfer)
- VALID: dex_amm, dex_orderbook, dex_aggregator, dex_clmm, liquidity_pools,
  yield_vaults, money_markets, collateralized_lending, liquidation_systems,
  flash_loan_systems, fixed_term_lending, bridge_protocols, fast_bridge_systems,
  staking_pools, oracle_feeds, proposal_systems, voting_systems, gauge_systems,
  nft_marketplaces, nft_minting_systems, perpetual_protocols, futures_markets,
  options_protocols, margin_trading_systems, blockchain_data, rollup_systems,
  account_abstraction, contract_registries, token_contracts, token_factories,
  stablecoin_systems, synthetic_asset_pools, reward_systems, fee_systems,
  domain_registries, attestation_systems, social_graphs, prediction_markets,
  pool_factories, vault_systems, limit_order_systems, keeper_networks,
  multisig_wallet, privacy_pools, bonding_curve_systems, rwa_protocols

────────────────────────────────────────────
## G3 RULES
- Atomic event-level action (one state transition)
- verb_first, lowercase + underscores
- Single action only — no compound events
- MUST NOT overlap with G2 meaning
- Reuse an existing G3 ONLY on a precise, specific match
- Create a NEW G3 whenever the existing registry lacks a precise fit
- Target: ~1 new G3 per 7 rows processed
- VALID: asset_exchange, liquidity_added, liquidity_removed,
  debt_origination, debt_repayment, collateral_seizure, flash_loan_taken,
  nft_minted, nft_transferred, royalty_distributed,
  vote_cast, proposal_created, proposal_executed,
  token_deposited, token_withdrawn, token_transferred, token_minted,
  reward_claimed, bridge_initiated, bridge_completed,
  stake_deposited, stake_withdrawn, price_updated, address_labeled,
  contract_initialized, pool_created, order_fulfilled, order_settled,
  rfq_filled, settlement_executed, yield_compounded

────────────────────────────────────────────
## OUTPUT FORMAT
Return ONLY a JSON array — one object per input row, EXACT same order, EXACT same count.
Each object: {{"namespace": "...", "table_name": "...", "g1": "...", "g2": "...", "g3": "..."}}
No markdown, no explanation, no preamble.
"""

HUMAN_PROMPT = """\
## EXISTING REGISTRY — reuse ONLY on precise match, do NOT stretch
G2 labels ({g2_used}/{g2_cap}, {g2_remaining} remaining):
{g2_list}

G3 labels ({g3_used}/{g3_cap}, {g3_remaining} remaining):
{g3_list}

────────────────────────────────────────────
## DIVERSITY BUDGET
Rows in this batch:             {count}
Rows remaining after this batch: {rows_remaining}

G2 budget pressure: {g2_pressure}
  → You MUST introduce at least {min_new_g2} new G2 label(s) this batch (0 = reuse only).
  G2 = system architecture — coin carefully; do NOT collapse distinct systems into one label.

G3 budget pressure: {g3_pressure}
  → You MUST introduce at least {min_new_g3} new G3 label(s) this batch.
  G3 = atomic actions — coin liberally; distinct event signatures almost always warrant distinct labels.
  OVERRIDE: Even if G3 pressure says OVER-LABELING, you still MUST add the mandated minimum.
{stall_override}
────────────────────────────────────────────
## INPUT — return exactly {count} objects in this exact order
{batch_json}
"""


# ─── REGISTRY ────────────────────────────────────────────────────────────────

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

    @property
    def g2_count(self) -> int:
        return len(self.g2_set)

    @property
    def g3_count(self) -> int:
        return len(self.g3_set)

    def register(self, g2: str, g3: str) -> None:
        if self.g2_count < G2_CAP or g2 in self.g2_set:
            self.g2_set.add(g2)
        if self.g3_count < G3_CAP or g3 in self.g3_set:
            self.g3_set.add(g3)

    def to_prompt_parts(self, count: int, rows_remaining: int, stall_g3: int = 0) -> dict:
        g2_remaining = max(0, G2_CAP - self.g2_count)
        g3_remaining = max(0, G3_CAP - self.g3_count)
        batch_fraction = count / max(rows_remaining, count)
        min_new_g2 = min(max(0, int(g2_remaining * batch_fraction)), g2_remaining)
        # G3 floor: always at least 1 when cap not reached, boosted when stalling
        base_g3 = min(max(1, round(g3_remaining * batch_fraction)), g3_remaining)
        if stall_g3 >= STALL_THRESHOLD:
            min_new_g3 = min(max(base_g3, 3), g3_remaining)
        else:
            min_new_g3 = max(base_g3, 1) if g3_remaining > 0 else 0

        g2_fill  = self.g2_count / G2_CAP
        g3_fill  = self.g3_count / G3_CAP
        row_done = max(0.01, 1.0 - (rows_remaining / max(rows_remaining + count, 1)))

        def pressure(fill: float, done: float) -> str:
            ratio = fill / max(done, 0.01)
            if ratio > 1.5:
                return "OVER-LABELING — consolidate similar systems, prefer reuse"
            elif ratio < 0.5:
                return "UNDER-LABELING — create new labels more aggressively"
            return "on track"

        stall_override = ""
        if stall_g3 >= STALL_THRESHOLD and g3_remaining > 0:
            stall_override = (
                f"\n⚠ STALL DETECTED: {stall_g3} consecutive batches with zero new G3 labels. "
                f"You MUST create at least {min_new_g3} new G3 label(s) in this batch. "
                f"Do NOT reuse existing G3 for actions with distinct semantics.\n"
            )

        return {
            "g2_list":        ", ".join(sorted(self.g2_set)) or "(empty — coin freely)",
            "g3_list":        ", ".join(sorted(self.g3_set)) or "(empty — coin freely)",
            "g2_used":        self.g2_count,
            "g2_cap":         G2_CAP,
            "g2_remaining":   g2_remaining,
            "g3_used":        self.g3_count,
            "g3_cap":         G3_CAP,
            "g3_remaining":   g3_remaining,
            "min_new_g2":     min_new_g2,
            "min_new_g3":     min_new_g3,
            "rows_remaining": max(0, rows_remaining - count),
            "g2_pressure":    pressure(g2_fill, row_done),
            "g3_pressure":    pressure(g3_fill, row_done),
            "stall_override": stall_override,
        }


# ─── UTILS ───────────────────────────────────────────────────────────────────

def _closest_existing(label: str, existing: set[str], fallback: str) -> str:
    """Semantic closest-match via bigram Jaccard similarity. Never blind alphabetical sort."""
    if not existing:
        return fallback

    def bigrams(s: str) -> set[str]:
        tokens = s.split("_")
        return set(tokens) | {f"{a}_{b}" for a, b in zip(tokens, tokens[1:])}

    query_bg = bigrams(label)
    best, best_score = fallback, -1.0
    for candidate in existing:
        cand_bg = bigrams(candidate)
        union   = query_bg | cand_bg
        score   = len(query_bg & cand_bg) / len(union) if union else 0.0
        if score > best_score:
            best, best_score = candidate, score
    return best


def _strip_fences(raw: str) -> str:
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def make_llm() -> ChatOpenRouter:
    return ChatOpenRouter(
        model="moonshotai/kimi-k2",
        temperature=0,
        openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
    )


# ─── LLM CALL ────────────────────────────────────────────────────────────────

def classify_batch(
    llm: ChatOpenRouter,
    registry: TaxonomyRegistry,
    rows: list[dict],
    rows_remaining: int,
    stall_g3: int = 0,
) -> list[dict]:
    """
    Call LLM for a batch. Returns list positionally aligned with `rows`.
    namespace/table_name always overwritten from `rows` — never trust LLM for these.
    Raises on JSON parse failure — retry loop handles it.
    """
    batch_items = [
        {"namespace": r["namespace"], "table_name": r["table_name"]}
        for r in rows
    ]
    batch_json    = json.dumps(batch_items, indent=2)
    prompt_parts  = registry.to_prompt_parts(len(rows), rows_remaining, stall_g3=stall_g3)
    system_text   = SYSTEM_PROMPT.format(g2_cap=G2_CAP, g3_cap=G3_CAP)
    human_text    = HUMAN_PROMPT.format(batch_json=batch_json, count=len(rows), **prompt_parts)

    response   = llm.invoke([SystemMessage(content=system_text), HumanMessage(content=human_text)])
    raw        = _strip_fences(response.content.strip())
    classified = json.loads(raw)

    # Positional alignment — overwrite namespace/table_name from ground truth
    aligned: list[dict] = []
    for i, original in enumerate(rows):
        item = classified[i] if i < len(classified) else {}
        if i >= len(classified):
            print(
                f"  WARN: LLM returned {len(classified)} items for {len(rows)} rows "
                f"— filling position {i} with fallback",
                file=sys.stderr,
            )
        item["namespace"]  = original["namespace"]
        item["table_name"] = original["table_name"]
        aligned.append(item)

    if len(classified) > len(rows):
        print(
            f"  WARN: LLM returned {len(classified)} items for {len(rows)} rows "
            f"— extra items discarded",
            file=sys.stderr,
        )

    return aligned


# ─── VALIDATION ──────────────────────────────────────────────────────────────

def validate_and_fix(item: dict, registry: TaxonomyRegistry) -> dict:
    """
    Validate G1/G2/G3 against registry caps.
    Cap overflow → semantic closest-match, logged as WARN.
    Cap-replaced labels are NOT registered — only LLM output gets registered.
    """
    ns  = item.get("namespace",  "")
    tbl = item.get("table_name", "")
    g1  = str(item.get("g1", FALLBACK_G1)).lower().strip()
    g2  = str(item.get("g2", FALLBACK_G2)).lower().strip()
    g3  = str(item.get("g3", FALLBACK_G3)).lower().strip()

    # G1: hard enum, no recovery possible
    if g1 not in G1_ENUM:
        print(f"  WARN: invalid G1 '{g1}' for {ns}.{tbl} → '{FALLBACK_G1}'", file=sys.stderr)
        g1 = FALLBACK_G1

    # G2 cap: new label at cap → closest existing, do NOT register
    was_g2_replaced = False
    if g2 not in registry.g2_set and registry.g2_count >= G2_CAP:
        replacement = _closest_existing(g2, registry.g2_set, FALLBACK_G2)
        print(f"  WARN: G2 cap — '{g2}' → '{replacement}' for {ns}.{tbl}", file=sys.stderr)
        g2 = replacement
        was_g2_replaced = True

    # G3 cap: same logic
    was_g3_replaced = False
    if g3 not in registry.g3_set and registry.g3_count >= G3_CAP:
        replacement = _closest_existing(g3, registry.g3_set, FALLBACK_G3)
        print(f"  WARN: G3 cap — '{g3}' → '{replacement}' for {ns}.{tbl}", file=sys.stderr)
        g3 = replacement
        was_g3_replaced = True

    # Only register labels that came from the LLM, not cap-fallback replacements
    if not was_g2_replaced and not was_g3_replaced:
        registry.register(g2, g3)
    elif not was_g2_replaced:
        if registry.g2_count < G2_CAP or g2 in registry.g2_set:
            registry.g2_set.add(g2)
    elif not was_g3_replaced:
        if registry.g3_count < G3_CAP or g3 in registry.g3_set:
            registry.g3_set.add(g3)

    return {"namespace": ns, "table_name": tbl, "g1": g1, "g2": g2, "g3": g3}


# ─── MAIN LOOP ───────────────────────────────────────────────────────────────

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

    with open(input_path, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    print(f"Total rows: {len(all_rows)}", file=sys.stderr)

    done_keys: set[tuple[str, str]] = set()
    if resume and Path(output_path).exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_keys.add((row["namespace"], row["table_name"]))
        print(f"Resuming — skipping {len(done_keys)} rows", file=sys.stderr)

    pending = [r for r in all_rows if (r["namespace"], r["table_name"]) not in done_keys]
    print(f"Rows to classify: {len(pending)}", file=sys.stderr)

    mode = "a" if resume and Path(output_path).exists() else "w"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_file = open(output_path, mode, newline="", encoding="utf-8")
    writer   = csv.DictWriter(out_file, fieldnames=["namespace", "table_name", "g1", "g2", "g3"])
    if mode == "w":
        writer.writeheader()

    stall_g2 = 0
    stall_g3 = 0

    try:
        for batch_start in range(0, len(pending), batch_size):
            batch          = pending[batch_start : batch_start + batch_size]
            batch_end      = batch_start + len(batch)
            rows_remaining = len(pending) - batch_start

            print(
                f"Classifying rows {batch_start + 1}–{batch_end} / {len(pending)} "
                f"| G2: {registry.g2_count}/{G2_CAP} "
                f"| G3: {registry.g3_count}/{G3_CAP}",
                file=sys.stderr,
            )

            # ── Expected vs actual coverage check ─────────────────────────
            if batch_start > 0:
                row_done_frac = batch_start / len(pending)
                expected_g2   = row_done_frac * G2_CAP
                expected_g3   = row_done_frac * G3_CAP
                g2_ratio      = registry.g2_count / max(expected_g2, 1)
                g3_ratio      = registry.g3_count / max(expected_g3, 1)
                if g2_ratio > 1.5:
                    print(
                        f"  WARN: G2 over-labeling "
                        f"({registry.g2_count} actual vs {expected_g2:.0f} expected)",
                        file=sys.stderr,
                    )
                if g2_ratio < 0.5:
                    print(
                        f"  WARN: G2 under-labeling "
                        f"({registry.g2_count} actual vs {expected_g2:.0f} expected)",
                        file=sys.stderr,
                    )
                if g3_ratio < 0.5:
                    print(
                        f"  WARN: G3 under-labeling "
                        f"({registry.g3_count} actual vs {expected_g3:.0f} expected)",
                        file=sys.stderr,
                    )

            # ── LLM call with retry ────────────────────────────────────────
            prev_g2    = registry.g2_count
            prev_g3    = registry.g3_count
            classified = None

            for attempt in range(3):
                try:
                    classified = classify_batch(llm, registry, batch, rows_remaining, stall_g3=stall_g3)
                    break
                except Exception as e:
                    print(f"  Attempt {attempt + 1} failed: {e}", file=sys.stderr)
                    if attempt < 2:
                        time.sleep(2 ** attempt)

            if classified is None:
                print(f"  All retries exhausted — writing fallback rows", file=sys.stderr)
                classified = [
                    {"namespace": r["namespace"], "table_name": r["table_name"],
                     "g1": FALLBACK_G1, "g2": FALLBACK_G2, "g3": FALLBACK_G3}
                    for r in batch
                ]

            # ── Validate and write atomically ──────────────────────────────
            buffer: list[dict] = []
            for item in classified:
                buffer.append(validate_and_fix(item, registry))

            for row in buffer:
                writer.writerow(row)
            out_file.flush()
            registry.save()

            # ── Stall detection (G2 and G3 tracked independently) ──────────
            if registry.g2_count == prev_g2:
                stall_g2 += 1
                if stall_g2 >= STALL_THRESHOLD:
                    print(
                        f"  WARN: {stall_g2} consecutive zero-G2 batches — "
                        f"LLM is over-collapsing G2.",
                        file=sys.stderr,
                    )
            else:
                stall_g2 = 0

            if registry.g3_count == prev_g3:
                stall_g3 += 1
                if stall_g3 >= STALL_THRESHOLD:
                    print(
                        f"  WARN: {stall_g3} consecutive zero-G3 batches — "
                        f"LLM is over-collapsing G3.",
                        file=sys.stderr,
                    )
            else:
                stall_g3 = 0

    finally:
        out_file.close()

    print(
        f"\nDone.\n"
        f"Output:    {output_path}\n"
        f"G2 unique: {registry.g2_count}/{G2_CAP}\n"
        f"G3 unique: {registry.g3_count}/{G3_CAP}",
        file=sys.stderr,
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify blockchain source tables into G1/G2/G3 taxonomy via LLM."
    )
    parser.add_argument("--input",      default=DEFAULT_INPUT)
    parser.add_argument("--output",     default=DEFAULT_OUTPUT)
    parser.add_argument("--registry",   default=DEFAULT_REGISTRY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--resume",     action="store_true")
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