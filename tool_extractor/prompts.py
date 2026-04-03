STAGE1_PROMPT = """
You are an expert onchain SQL analyst.

You will receive a raw Dune query (title + SQL). Your job is to decide if it
represents one logical tool or multiple independent tools.

## Rules

- If all CTEs feed into a single final SELECT → one tool, return as-is
- If the query contains multiple independent SELECT statements that could each
  run standalone and return meaningful data → split into separate units
- If a CTE is only meaningful as an intermediate step for the outer query →
  do not split it out
- Never discard. If a query is trivial, still return it as one unit.

## Output format

Return ONLY valid JSON. No markdown, no explanation, no preamble.

{
  "units": [
    {
      "label": "short descriptive label for this unit",
      "sql": "the SQL for this unit"
    }
  ]
}

## Query to decompose

Title: {{title}}

SQL:
{{sql}}
"""


STAGE2_PROMPT = """
You are an expert onchain data analyst and taxonomy engineer.

You will receive a SQL unit (already decomposed by Stage 1). Classify it into
the grimoire taxonomy and extract its inputs.

## Taxonomy rules

### g1 — domain
Must exactly match one domain from the registry:
{{domains}}

### g2 — category
Must come from the controlled vocabulary below. If none fit, use the closest.

Controlled g2 vocabulary per domain:

Protocols:     Trading Activity | Pool Analytics | Liquidity Analytics | Fee Analytics |
               Yield Analytics | Lending Activity | Staking Operations | Launch Analytics |
               Usage Metrics | Payment Analytics | Vault Analytics | Token Lifecycle |
               Execution Analytics | Competitive Analysis | User Analytics | Revenue Analytics

Tokens:        Trading Activity | Holder Analytics | Supply Analytics | Price Analytics |
               Transfer Analytics | Token Discovery | DeFi Usage | Cross-chain Analytics |
               Social Trading | Opportunity Screening | Tokenomics | Flow Analytics

Wallets:       Trading Performance | Transaction History | Holder Analytics |
               Trading Intelligence | Trade Attribution | Contract Analytics |
               Balance Tracking | Contract Metadata | NFT Holdings | Token Deployer Analysis |
               Protocol User Analysis

Chains:        Activity Metrics | Economic Metrics | DeFi Metrics | Ecosystem Analytics

Network:       Activity Metrics | Fee Metrics

DAOs:          Governance Activity

Forensics:     Fund Flow | Contract Labels | Trace Monitoring | Transaction Inspection |
               Contract Discovery | Treasury Monitoring | Data Inspection

Derivatives:   Market Participation | Yield Analytics

### g3 — tool identity
THE most important field. Canonical tool name in the Power Toolbox.

Rules:
- Max 4 words
- NEVER contains a protocol name, chain name, or token name
- Reusable across any protocol, chain, or token
- Append a structural suffix that encodes output type:
  - Series     — time series (daily, weekly, hourly trends)
  - Snapshot   — point-in-time (current state, latest values)
  - Leaderboard — ranked list (top N by metric)
  - Tracker    — ongoing state changes, events
  - Screener   — items matching filter criteria
  - Lookup     — metadata resolution (address → name)
  - Feed       — raw event stream (logs, transfers, txs)
  - Monitor    — threshold-based or anomaly detection
- Must be reusable across 50+ queries — if too specific, generalize

### g4 — variant
What is specifically measured within this g3. Max 4 words. No protocol names.

### g5 — provenance
Concrete description of the source query. May name protocols/chains/tokens.
Max 10 words. Never shown to users.

## Registry match rules

Check the registry for a matching g3 before creating a new tool.

- Same g3 + compatible inputs  → match: true, set tool_id
- Same g3 + different inputs   → match: true, set tool_id, note: "extends inputs"
- No match                     → match: false, tool_id: null

Two g3s are the same if they answer the same question for different protocols
or chains. When in doubt, merge.

## Input extraction rules

Only extract runtime parameters a user fills in:
- Onchain identifiers: wallet, contract, token addresses
- Selectors: token symbols, protocol names, chain names
- Time bounds: start date, end date, lookback window, granularity
- Numeric filters: min/max amounts, top N, thresholds

Do NOT extract:
- SQL table names, schema names, CTE names
- Protocol-internal constants with no user-facing meaning
- SQL operators or function names

Dune {{param}} template params are always real inputs.

Hardcoded values → lift as input, set as default. Never leave default: null
when a hardcoded value exists in the SQL.

No hardcoded value → default: null.
NEVER use "query-specific" or descriptive text as a default.

Max 4 inputs per tool. Keep the most analytically meaningful ones.

Input schema:
{
  "key": "snake_case",
  "label": "Human readable label",
  "type": "address|token|protocol|chain|date_range|number|text",
  "required": true,
  "default": null
}

required: true  → no sensible default exists, user must provide
required: false → default exists, tool runs without user input

## Output format

Return ONLY valid JSON. No markdown, no explanation, no preamble.

{
  "match": false,
  "tool_id": null,
  "note": null,
  "g1": "Protocols",
  "g2": "Yield Analytics",
  "g3": "Rate Time Series",
  "g4": "Borrow Supply APY",
  "g5": "Aave V3 Sonic daily supply and borrow APY by asset",
  "inputs": [
    {
      "key": "protocol",
      "label": "Protocol",
      "type": "protocol",
      "required": false,
      "default": "aave_v3"
    },
    {
      "key": "chain",
      "label": "Chain",
      "type": "chain",
      "required": false,
      "default": "sonic"
    },
    {
      "key": "token_symbol",
      "label": "Asset symbol",
      "type": "token",
      "required": true,
      "default": null
    },
    {
      "key": "lookback_days",
      "label": "Lookback days",
      "type": "number",
      "required": false,
      "default": "365"
    }
  ]
}

## Current grimoire registry

{{registry_json}}

## SQL unit to classify

Label: {{label}}

SQL:
{{sql}}
"""

# ─── UTILS ───────────────────────────────────────────────────────────────────

def build_stage1_prompt(title: str, sql: str) -> str:
    return STAGE1_PROMPT \
        .replace("{{title}}", title) \
        .replace("{{sql}}", sql)


def build_stage2_prompt(
    label: str,
    sql: str,
    registry_json: str,
    domains: list[dict],
) -> str:
    domain_list = "\n".join([
        f"  - {d['name']}: {d['description']}"
        for d in domains
    ])
    return STAGE2_PROMPT \
        .replace("{{domains}}", domain_list) \
        .replace("{{registry_json}}", registry_json) \
        .replace("{{label}}", label) \
        .replace("{{sql}}", sql)