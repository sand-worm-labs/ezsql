"""
Batch-extract canonical Grimoire tools from DB queries via OpenRouter.

Tables are fetched from popular_tables (DB) in impact order (top_80 tier).
Pass 1: single-table queries  — per table, parallel workers.
Pass 2: multi-table (join)    — per table, parallel workers.
Pass 3: cross-domain composites — deliberate namespace pairs, seeded with
         existing tools as building blocks. This is where whale_accumulation_
         detector-class tools come from. Do not skip it.

Seen tracking: grimoire_extract_seen + grimoire_seen_tables DB tables.

Usage:
    python scripts/batch_extract.py [--batch 1000] [--workers 5] [--model MODEL]
    python scripts/batch_extract.py --pass3-only   # only run composite pass
"""

import os
import sys
import json
import re
import time
import argparse
import textwrap
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy import create_engine, text as sa_text

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.connection import get_session
from tool_extractor.registry import get_registry_json

ROOT        = Path(__file__).parent.parent
POPULAR_CSV = ROOT / "popular_tables_with_impact.csv"

TOKEN_LIMIT = 800_000   # max tokens per LLM batch
COMPOSITE_QUERY_LIMIT = 40  # queries pulled per table in a composite batch


# ═══════════════════════════════════════════════════════════════════════════════
# ⬢ PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Standard extraction prompt (pass 1 + pass 2) ─────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a blockchain analytics tool extractor for the Grimoire registry.

    MISSION: Extract REUSABLE tools that provide insights across any protocol, chain,
    or token — not one-off reports for a specific project.

    SKIP — do not extract:
      - Simple aggregations (SELECT SUM/COUNT with one table and no join)
      - Basic lookups (SELECT * WHERE address = ...)
      - Queries whose only value is a specific protocol/token/chain name
        (e.g. "top Uniswap pools yesterday", "BTC returns by president")
      - Single-table queries unless they reveal a non-obvious reusable insight

    EXTRACT (high or medium complexity only):
      - Multi-table joins with meaningful logic
      - Whale detection and accumulation patterns
      - Bot / wash-trading classification
      - Protocol health scoring
      - Cross-protocol or cross-chain analytics
      - Cohort analysis, retention, new-vs-returning
      - Anomaly detection, large-trade alerts

    GENERICITY IS MANDATORY:
      - g3 MUST NOT contain any protocol, chain, token, or project name
        ✅ dex_volume_by_protocol    ❌ sushi_tokens_ohlcv
        ✅ chain_tvl_breakdown       ❌ pools_tvl_arbitrum
        ✅ token_price_returns       ❌ btc_returns_by_president
        ✅ ramp_volume_tracker       ❌ monerium_ramp_volume
      - description MUST NOT name any protocol, chain, or token
      - If a query is Uniswap-only but the technique is generic, extract it as
        scope:protocol:uniswap — still no protocol name in g3

    TAXONOMY:
      g1 = namespace   (defi|token|lending|mev|infra|price|perp|nft|prediction|social)
      g2 = category    (dex|lp|holders|oracle|wallet|gas|identity|borrow|sandwich|arb)
                       create a new category if none fits
      g3 = tool name   snake_case, describes WHAT the insight is — never WHO provides it

    COMPLEXITY FILTER:
      high   = multi-join, window functions, CTEs with logic, scoring, classification
      medium = two-table join with non-trivial filter or aggregation
      low    = single-table count/sum — SKIP entirely unless insight is non-obvious

    COLLAPSE RULES:
      - Queries differing only by WHERE clause → one tool + param
      - Protocol = param unless source table is unique to that protocol
      - chain='solana' is a param, not a separate tool
      - Token symbol = param, not a separate tool

    INPUTS: extract user-facing runtime parameters the query actually needs.
      param types: chain | protocol | date_from | date_to | interval | top_n |
                   min_usd | contract_address | wallet_address | pool_address |
                   token_symbol | lookback_days | threshold_usd | text | number
      - Dune {{param}} syntax → required: true, default: null
      - Hardcoded literal    → required: false, default: that literal
      - Never extract table names, schema names, CTE aliases, SQL keywords
      - Max 6 inputs per tool

    OUTPUT: one Markdown section per tool, separated by "##". No JSON, no explanation.

    ## <g3>
    g1: <value>
    g2: <value>
    g3: <value>
    g4: <metric label, max 4 words>
    g5: <provenance, max 10 words>
    description: <Verb-first. No protocol/chain/token names.>
    scope: generic | protocol:NAME
    viz: timeseries | table | metric | leaderboard | heatmap
    returns: col:type, col:type
    inputs:
      - key:<k> label:<l> type:<t> required:true|false default:<val or null>
    source_query_ids: id1, id2, id3

    SCOPE: generic = works for any protocol. protocol:X = source table is X-specific.
    RETURNS: output columns as name:type. type = string | number | boolean.
    VIZ: timeseries if output has a time column. leaderboard if ranked by value.
         metric if a single scalar. table otherwise.
""")


# ── Composite extraction prompt (pass 3) ─────────────────────────────────────

COMPOSITE_PROMPT = textwrap.dedent("""\
    You are a blockchain analytics tool extractor for the Grimoire registry.
    This is a COMPOSITE GENERATION pass. Your only job is to extract tools
    that combine signals from 2 or more namespaces into a single higher-order insight.

    WHAT A COMPOSITE TOOL IS:
      - source_query_ids must span at least 2 different table namespaces
      - The combined insight must be richer than either input alone
      - Examples of legitimate composites:
          whale_accumulation_detector   = wallet balances + price timeseries
          price_volume_divergence       = dex volume + price feed
          protocol_health_score         = DAU + volume + token price + retention
          wash_trading_scorer           = circular transfers + wallet clustering
          wallet_behavior_classifier    = gas patterns + timing + tx frequency
          liquidation_cascade_detector  = borrow positions + price drops
          nft_wash_trade_detector       = nft transfers + price + wallet graph

    BUILDING BLOCKS:
      Below you will receive a list of existing registry tools. You MAY reference
      these conceptually — your new composite tool combines their signals even if
      the source queries overlap. Do NOT re-emit a tool that already exists
      (same g1+g3). Do NOT emit a tool that is just a rename of an existing one.

    COMPLEXITY: ALL outputs must be complexity=high. If you cannot produce a
    genuinely composite high-complexity tool from the provided queries, output
    nothing. Do not pad with medium or low tools.

    MANDATORY: For each composite tool you emit, add a line:
      combines: <tool_id_1>, <tool_id_2>   (existing tool IDs being combined)

    TAXONOMY + OUTPUT FORMAT: identical to standard extraction.

    ## <g3>
    g1: <value>
    g2: <value>
    g3: <value>
    g4: <metric label, max 4 words>
    g5: <provenance, max 10 words>
    description: <Verb-first. No protocol/chain/token names.>
    scope: generic | protocol:NAME
    viz: timeseries | table | metric | leaderboard | heatmap
    returns: col:type, col:type
    combines: existing_tool_id_1, existing_tool_id_2
    inputs:
      - key:<k> label:<l> type:<t> required:true|false default:<val or null>
    source_query_ids: id1, id2, id3

    If you cannot find any genuine composites in this batch, respond with exactly:
      NO_COMPOSITES
""")


# ═══════════════════════════════════════════════════════════════════════════════
# ⬢ COMPOSITE PAIRS  (pass 3 cross-domain batches)
# ═══════════════════════════════════════════════════════════════════════════════

# Each entry is a list of (namespace, table) tuples to pull queries from.
# These pairs are chosen because their signals compose into meaningful higher-order tools.
COMPOSITE_PAIRS: list[tuple[str, list[tuple[str, str]]]] = [
    ("whale_accumulation",   [("prices",  "usd"),          ("tokens",  "transfers"),     ("tokens",  "balances")]),
    ("price_volume_div",     [("prices",  "usd"),          ("dex",     "trades")]),
    ("protocol_health",      [("dex",     "trades"),       ("prices",  "usd"),            ("tokens",  "transfers")]),
    ("wallet_classifier",    [("tokens",  "transfers"),    ("gas",     "fees"),           ("transactions", "v2")]),
    ("wash_trading",         [("nft",     "trades"),       ("prices",  "usd"),            ("tokens",  "transfers")]),
    ("liquidation_cascade",  [("aave",    "borrows"),      ("prices",  "usd")]),
    ("mev_price_impact",     [("dex",     "trades"),       ("sandwich","transactions")]),
    ("lending_health",       [("aave",    "borrows"),      ("aave",    "repays"),         ("prices",  "usd")]),
    ("nft_wash_detector",    [("nft",     "trades"),       ("tokens",  "transfers")]),
    ("cross_chain_arb",      [("dex",     "trades"),       ("bridge",  "transfers")]),
    ("retention_revenue",    [("dex",     "trades"),       ("tokens",  "balances"),       ("prices",  "usd")]),
    ("bot_vs_human",         [("transactions", "v2"),      ("gas",     "fees"),           ("dex",     "trades")]),
]


# ═══════════════════════════════════════════════════════════════════════════════
# ⬢ DB BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_tables(engine):
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS grimoire_tools (
            tool_id        TEXT PRIMARY KEY,
            g1             TEXT NOT NULL,
            g2             TEXT NOT NULL,
            g3             TEXT NOT NULL,
            g4             TEXT NOT NULL DEFAULT '',
            g5             TEXT NOT NULL DEFAULT '',
            description    TEXT NOT NULL,
            scope          TEXT NOT NULL DEFAULT 'generic',
            returns        JSONB NOT NULL DEFAULT '[]',
            viz            TEXT NOT NULL DEFAULT 'table',
            inputs         JSONB NOT NULL DEFAULT '[]',
            source_queries TEXT[] NOT NULL DEFAULT '{}',
            combines       TEXT[] NOT NULL DEFAULT '{}',
            usage_count    INTEGER NOT NULL DEFAULT 0,
            last_used_at   TIMESTAMPTZ,
            created_at     TIMESTAMPTZ DEFAULT NOW(),
            updated_at     TIMESTAMPTZ DEFAULT NOW(),
            embedding      vector(1536)
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_grimoire_g1_g3 ON grimoire_tools (g1, g3)",
        "CREATE INDEX IF NOT EXISTS idx_grimoire_g1_g2 ON grimoire_tools (g1, g2)",
        "CREATE INDEX IF NOT EXISTS idx_grimoire_combines ON grimoire_tools USING gin(combines)",
        """
        CREATE TABLE IF NOT EXISTS grimoire_extract_seen (
            query_id TEXT PRIMARY KEY,
            seen_at  TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS grimoire_seen_tables (
            full_name TEXT PRIMARY KEY,
            seen_at   TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS grimoire_seen_composite_pairs (
            pair_key TEXT PRIMARY KEY,
            seen_at  TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        # separate execute — psycopg2 drops subsequent statements in multi-statement calls
        "ALTER TABLE grimoire_tools ADD COLUMN IF NOT EXISTS combines TEXT[] NOT NULL DEFAULT '{}'",
    ]
    with engine.begin() as conn:
        for stmt in ddl_statements:
            conn.execute(sa_text(stmt))


# ═══════════════════════════════════════════════════════════════════════════════
# ⬢ SEEN TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

def load_seen(engine) -> set[str]:
    with engine.connect() as conn:
        a = conn.execute(sa_text("SELECT query_id FROM grimoire_extract_seen")).fetchall()
        b = conn.execute(sa_text("SELECT unnest(source_queries) FROM grimoire_tools")).fetchall()
    seen: set[str] = set()
    seen.update(r[0] for r in a)
    seen.update(str(r[0]) for r in b)
    return seen


def load_seen_tables(engine) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(sa_text("SELECT full_name FROM grimoire_seen_tables")).fetchall()
    return {r[0] for r in rows}


def load_seen_composite_pairs(engine) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(sa_text("SELECT pair_key FROM grimoire_seen_composite_pairs")).fetchall()
    return {r[0] for r in rows}


def mark_seen(engine, ids: list[str]):
    if not ids:
        return
    with engine.begin() as conn:
        conn.execute(
            sa_text("INSERT INTO grimoire_extract_seen (query_id) VALUES (:qid) ON CONFLICT DO NOTHING"),
            [{"qid": i} for i in ids],
        )


def mark_table_seen(engine, full_name: str):
    with engine.begin() as conn:
        conn.execute(sa_text(
            "INSERT INTO grimoire_seen_tables (full_name) VALUES (:n) ON CONFLICT DO NOTHING"
        ), {"n": full_name})


def mark_composite_pair_seen(engine, pair_key: str):
    with engine.begin() as conn:
        conn.execute(sa_text(
            "INSERT INTO grimoire_seen_composite_pairs (pair_key) VALUES (:k) ON CONFLICT DO NOTHING"
        ), {"k": pair_key})


# ═══════════════════════════════════════════════════════════════════════════════
# ⬢ POPULAR TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def load_popular_tables() -> list[dict]:
    if not POPULAR_CSV.exists():
        print(f"ERROR: {POPULAR_CSV} not found", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(POPULAR_CSV)
    df = df[df["tier"] == "top_80"].reset_index(drop=True)
    if df.empty:
        print("ERROR: no top_80 rows in CSV", file=sys.stderr)
        sys.exit(1)
    return [
        {"namespace": str(r["namespace"]), "table": str(r["table"]), "full_name": r["full_name"]}
        for _, r in df.iterrows()
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# ⬢ QUERY FETCHERS
# ═══════════════════════════════════════════════════════════════════════════════

_TABLE_COUNT = """
    (
        SELECT COUNT(DISTINCT m[1])
        FROM regexp_matches(
            query_sql,
            '(?:FROM|JOIN)\\s+("?[\\w]+"?\\."?[\\w]+"?(?:\\."?[\\w]+"?)?)',
            'gi'
        ) AS m
    )
"""


def _tpat(ns: str, tbl: str) -> str:
    return f'(?:FROM|JOIN)\\s+"?{re.escape(ns)}"?\\."?{re.escape(tbl)}"?'


def fetch_single(engine, ns: str, tbl: str, seen: set[str]) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(f"""
            SELECT query_id::text, name, query_sql
            FROM "queries"
            WHERE query_sql ~* %s
              AND query_sql IS NOT NULL
              AND LENGTH(query_sql) <= 1000
              AND {_TABLE_COUNT} = 1
        """, (_tpat(ns, tbl),)).fetchall()
    return [{"query_id": r[0], "name": r[1], "query_sql": r[2]}
            for r in rows if r[0] not in seen]


def fetch_multi(engine, ns: str, tbl: str, seen: set[str]) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(f"""
            SELECT query_id::text, name, query_sql
            FROM "queries"
            WHERE query_sql ~* %s
              AND query_sql IS NOT NULL
              AND LENGTH(query_sql) <= 1000
              AND {_TABLE_COUNT} > 1
        """, (_tpat(ns, tbl),)).fetchall()
    return [{"query_id": r[0], "name": r[1], "query_sql": r[2]}
            for r in rows if r[0] not in seen]


def fetch_for_composite(engine, ns: str, tbl: str,
                        seen: set[str], limit: int) -> list[dict]:
    """
    For composite batches: fetch multi-table queries touching this table,
    preferring unseen but also allowing seen (composites may reuse source queries
    to produce a higher-order tool).
    """
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(f"""
            SELECT query_id::text, name, query_sql
            FROM "queries"
            WHERE query_sql ~* %s
              AND query_sql IS NOT NULL
              AND {_TABLE_COUNT} > 1
            ORDER BY random()
            LIMIT %s
        """, (_tpat(ns, tbl), limit)).fetchall()
    # for composites we allow seen queries — we want cross-namespace signal
    return [{"query_id": r[0], "name": r[1], "query_sql": r[2]} for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# ⬢ TOKEN ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def est_tokens(q: dict) -> int:
    return (
        len(q.get("query_id") or "") +
        len(q.get("name") or "") +
        len(q.get("query_sql") or "")
    ) // 2


# ═══════════════════════════════════════════════════════════════════════════════
# ⬢ MESSAGE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _registry_ids(registry_json: str) -> str:
    try:
        tools = json.loads(registry_json)
        ids   = [f"{t['g1']}.{t['g3']}" for t in tools]
        return "## Existing tool IDs — do NOT duplicate\n" + "\n".join(ids) + "\n\n"
    except Exception:
        return ""


def build_message(queries: list[dict], registry_json: str) -> str:
    """Standard message for pass 1 + pass 2."""
    parts = [_registry_ids(registry_json), f"## Queries ({len(queries)})\n"]
    for q in queries:
        parts.append(f"[{q['query_id']}] {q.get('name','')}\n{(q.get('query_sql') or '').strip()}\n---")
    return "\n".join(parts)


def build_composite_message(queries: list[dict], registry_json: str,
                             seed_g1s: set[str]) -> str:
    """Composite message for pass 3 — same ID-only anti-dupe block."""
    parts = [_registry_ids(registry_json), f"## Source queries ({len(queries)})\n"]
    for q in queries:
        parts.append(f"[{q['query_id']}] {q.get('name','')}\n{(q.get('query_sql') or '').strip()}\n---")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# ⬢ PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_md_tools(md: str) -> list[dict]:
    """Parse LLM Markdown output into tool dicts. Tolerates truncation."""
    tools = []
    for section in re.split(r"\n##\s+", "\n" + md.strip()):
        section = section.strip()
        if not section:
            continue
        lines = section.splitlines()
        t: dict = {
            "inputs": [], "source_query_ids": [], "returns": [],
            "viz": "table", "scope": "generic", "combines": [],
        }
        in_inputs = False
        for line in lines:
            line = line.strip()
            if not line:
                in_inputs = False
                continue
            if line.startswith("inputs:"):
                in_inputs = True
                continue
            if in_inputs and line.startswith("-"):
                raw = line.lstrip("- ")
                inp: dict = {"required": True, "default": None}
                for part in re.split(r"\s+(?=\w+:)", raw):
                    if ":" in part:
                        k, _, v = part.partition(":")
                        v = v.strip()
                        if k == "required":
                            inp[k] = v.lower() == "true"
                        elif k == "default":
                            inp[k] = None if v.lower() == "null" else v
                        else:
                            inp[k] = v
                if "key" in inp:
                    t["inputs"].append(inp)
                continue
            in_inputs = False
            if ":" in line:
                k, _, v = line.partition(":")
                k, v = k.strip(), v.strip()
                if k == "source_query_ids":
                    t["source_query_ids"] = [x.strip() for x in v.split(",") if x.strip()]
                elif k == "combines":
                    t["combines"] = [x.strip() for x in v.split(",") if x.strip()]
                elif k == "returns":
                    t["returns"] = [x.strip() for x in v.split(",") if x.strip()]
                elif k in ("g1","g2","g3","g4","g5","description","scope","viz"):
                    t[k] = v
        if t.get("g1") and t.get("g3"):
            tools.append(t)
    return tools


# ═══════════════════════════════════════════════════════════════════════════════
# ⬢ DB WRITER
# ═══════════════════════════════════════════════════════════════════════════════

def write_tools(engine, tools: list[dict]) -> int:
    saved = 0
    for tool in tools:
        src      = [str(x) for x in (tool.pop("source_query_ids", []) or [])]
        combines = [str(x) for x in (tool.pop("combines", []) or [])]
        g1       = tool.get("g1", "").strip()
        g3       = tool.get("g3", "").strip()
        if not g1 or not g3:
            print("    ✗ missing g1/g3 — skip", file=sys.stderr)
            continue

        tid = f"{g1}.{g3}"
        with engine.begin() as conn:
            existing = conn.execute(sa_text(
                "SELECT tool_id FROM grimoire_tools WHERE g1=:g1 AND g3=:g3"
            ), {"g1": g1, "g3": g3}).mappings().first()

            if existing:
                print(f"    → merge '{existing['tool_id']}'")
                conn.execute(sa_text("""
                    UPDATE grimoire_tools SET
                        source_queries = COALESCE(
                            (SELECT array_agg(DISTINCT e)
                             FROM unnest(source_queries || CAST(:src AS text[])) AS e),
                            '{}'
                        ),
                        combines = COALESCE(
                            (SELECT array_agg(DISTINCT e)
                             FROM unnest(combines || CAST(:combines AS text[])) AS e),
                            '{}'
                        ),
                        updated_at = NOW()
                    WHERE tool_id = :tid
                """), {"src": src, "combines": combines, "tid": existing["tool_id"]})
            else:
                print(f"    + insert '{tid}'")
                conn.execute(sa_text("""
                    INSERT INTO grimoire_tools
                        (tool_id,g1,g2,g3,g4,g5,
                         description,scope,returns,viz,inputs,source_queries,combines)
                    VALUES
                        (:tid,:g1,:g2,:g3,:g4,:g5,
                         :desc,:scope,
                         CAST(:returns AS jsonb),:viz,
                         CAST(:inputs AS jsonb),
                         :src,:combines)
                    ON CONFLICT (g1,g3) DO NOTHING
                """), {
                    "tid":     tid,
                    "g1":      g1,
                    "g2":      tool.get("g2",""),
                    "g3":      g3,
                    "g4":      tool.get("g4",""),
                    "g5":      tool.get("g5",""),
                    "desc":    tool.get("description",""),
                    "scope":   tool.get("scope","generic"),
                    "returns": json.dumps(tool.get("returns",[])),
                    "viz":     tool.get("viz","table"),
                    "inputs":  json.dumps(tool.get("inputs",[])),
                    "src":     src,
                    "combines": combines,
                })
        saved += 1
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# ⬢ FLUSH
# ═══════════════════════════════════════════════════════════════════════════════

def flush(engine, llm, batch: list[dict], registry_json: str,
          seen: set[str], state: dict, lock: threading.Lock,
          system_prompt: str = SYSTEM_PROMPT,
          message_fn=None,
          retries: int = 50):
    if not batch:
        return
    with lock:
        state["n"] += 1
        batch_n = state["n"]

    print(f"\n[batch {batch_n}] {len(batch)} queries → LLM…")
    msg_builder = message_fn or (lambda b, r: build_message(b, r))

    for attempt in range(1, retries + 1):
        try:
            raw = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=msg_builder(batch, registry_json)),
            ]).content.strip()

            if raw == "NO_COMPOSITES":
                print("  → no composites found (LLM explicit)")
                return

            tools = parse_md_tools(raw)
            if not tools:
                print(f"  ✗ no tools parsed — raw[:200]:\n{raw[:200]}", file=sys.stderr)
                return

            print(f"  extracted {len(tools)}")
            n   = write_tools(engine, tools)
            ids = [r["query_id"] for r in batch]
            mark_seen(engine, ids)
            with lock:
                state["total"] += n
                seen.update(ids)
            print(f"  wrote {n}")
            return

        except Exception as e:
            if attempt < retries:
                detail = e.args[0] if e.args else ""
                print(f"  ✗ attempt {attempt}/{retries} [{type(e).__name__}]: {detail}", file=sys.stderr)
                time.sleep(10)
            else:
                print(f"  ✗ failed after {retries} attempts: {e}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════════
# ⬢ PASS RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_table_passes(engine, llm, tables: list[dict], seen: set[str],
                     seen_tables: set[str], registry_json: str,
                     batch_size: int, state: dict, lock: threading.Lock,
                     workers: int):
    """
    Pass 1 (single) + Pass 2 (multi) — one table at a time, parallel batches within.
    Loops until every table returns zero unseen rows.
    A table is only marked done when it returns 0 rows, not after one pass.
    """

    def make_batches(rows: list[dict]) -> list[list[dict]]:
        batches, sub, sub_tokens = [], [], 0
        for row in rows:
            sub.append(row)
            sub_tokens += est_tokens(row)
            if len(sub) >= batch_size or sub_tokens >= TOKEN_LIMIT:
                batches.append(sub[:])
                sub.clear(); sub_tokens = 0
        if sub:
            batches.append(sub)
        return batches

    def process_table(t: dict) -> int:
        """Drain one iteration of this table. Returns rows processed (0 = exhausted)."""
        single = fetch_single(engine, t["namespace"], t["table"], seen)
        multi  = fetch_multi(engine, t["namespace"], t["table"], seen)
        total  = len(single) + len(multi)
        if total == 0:
            return 0
        print(f"\n{t['full_name']}: {len(single)} single / {len(multi)} multi")
        batches = make_batches(single + multi)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(flush, engine, llm, b, registry_json, seen, state, lock)
                    for b in batches]
            for fut in as_completed(futs):
                if fut.exception():
                    print(f"  ✗ batch error: {fut.exception()}", file=sys.stderr)
        return total

    iteration = 0
    while True:
        iteration += 1
        pending = [t for t in tables if t["full_name"] not in seen_tables]
        if not pending:
            print(f"\nAll tables exhausted after {iteration - 1} iteration(s).")
            break

        print(f"\n{'─'*60}")
        print(f"Iteration {iteration} — {len(pending)} tables with unseen queries")
        print(f"{'─'*60}")

        newly_done = 0
        for t in pending:
            try:
                result = process_table(t)
                if result == 0:
                    mark_table_seen(engine, t["full_name"])
                    seen_tables.add(t["full_name"])
                    newly_done += 1
            except Exception as exc:
                print(f"  ✗ {t['full_name']}: {exc}", file=sys.stderr)

        print(f"Iteration {iteration} complete — {newly_done}/{len(pending)} tables exhausted")


def run_composite_pass(engine, llm, seen: set[str],
                       seen_pairs: set[str], registry_json: str,
                       state: dict, lock: threading.Lock):
    """
    Pass 3: cross-domain composite generation.
    For each COMPOSITE_PAIR, fetches queries from all member tables,
    mixes them into a single batch, and sends with the composite prompt
    seeded with existing tools from the relevant namespaces.
    """
    print(f"\n{'═'*60}")
    print("Pass 3: composite generation")
    print(f"{'═'*60}")

    for pair_name, table_specs in COMPOSITE_PAIRS:
        if pair_name in seen_pairs:
            print(f"  {pair_name}: skip (done)")
            continue

        # derive g1 namespaces for seed selection
        # best-effort: map table namespace to g1 (they often match)
        seed_g1s = {ns for ns, _ in table_specs}

        # collect queries from all tables in the pair
        batch: list[dict] = []
        for ns, tbl in table_specs:
            rows = fetch_for_composite(engine, ns, tbl, seen, COMPOSITE_QUERY_LIMIT)
            batch.extend(rows)

        if not batch:
            print(f"  {pair_name}: no queries found — skip")
            mark_composite_pair_seen(engine, pair_name)
            continue

        # deduplicate by query_id
        seen_ids: set[str] = set()
        deduped: list[dict] = []
        for r in batch:
            if r["query_id"] not in seen_ids:
                seen_ids.add(r["query_id"])
                deduped.append(r)

        print(f"\n  {pair_name}: {len(deduped)} queries from {len(table_specs)} tables")

        flush(
            engine, llm, deduped, registry_json, seen, state, lock,
            system_prompt=COMPOSITE_PROMPT,
            message_fn=lambda b, r, g=seed_g1s: build_composite_message(b, r, g),
        )
        mark_composite_pair_seen(engine, pair_name)


# ═══════════════════════════════════════════════════════════════════════════════
# ⬢ MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run(batch_size: int, model: str, temperature: float,
        workers: int, pass3_only: bool):

    llm    = ChatOpenRouter(model=model, temperature=temperature,
                            openrouter_api_key=os.environ["OPENROUTER_API_KEY"])
    engine = create_engine(os.environ["DATABASE_URL"], pool_size=workers + 4)

    ensure_tables(engine)

    seen        = load_seen(engine)
    seen_tables = load_seen_tables(engine)
    seen_pairs  = load_seen_composite_pairs(engine)
    reg_json    = get_registry_json()
    state       = {"n": 0, "total": 0}
    lock        = threading.Lock()

    print(f"model         : {model}")
    print(f"workers       : {workers}")
    print(f"seen query IDs: {len(seen)}")
    print(f"seen tables   : {len(seen_tables)}")
    print(f"seen pairs    : {len(seen_pairs)}")

    if not pass3_only:
        tables = load_popular_tables()
        print(f"top_80 tables : {len(tables)}\n")

        print("Pass 1 + 2: single and multi-table queries")
        run_table_passes(
            engine, llm, tables, seen, seen_tables,
            reg_json, batch_size, state, lock, workers,
        )
        # reload registry after passes 1+2 enrich it
        reg_json = get_registry_json()

    run_composite_pass(engine, llm, seen, seen_pairs, reg_json, state, lock)

    print(f"\nDone. {state['total']} tools written.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch",       type=int,   default=900)
    p.add_argument("--model",       type=str,   default="x-ai/grok-4.1-fast")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers",     type=int,   default=5)
    p.add_argument("--pass3-only",  action="store_true",
                   help="skip pass 1+2, run composite generation only")
    args = p.parse_args()
    run(args.batch, args.model, args.temperature, args.workers, args.pass3_only)