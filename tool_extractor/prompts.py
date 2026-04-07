STAGE2_PROMPT = """
You are a grimoire taxonomy engineer. Classify one SQL query into a reusable onchain analytics tool.

## THE ONE RULE THAT MATTERS
g3 is the merge key. ~150 unique g3s will cover 800K queries.
Before inventing a new g3, find the existing one that fits.
If you create a new g3 for every query, you are wrong.

## Domains (g1) — pick exactly one
{domains}



Chains:      Activity Metrics | Economic Metrics | DeFi Metrics | Ecosystem Analytics
Network:     Activity Metrics | Fee Metrics
DAOs:        Governance Activity
Forensics:   Fund Flow | Contract Labels | Trace Monitoring | Transaction Inspection |
             Contract Discovery | Treasury Monitoring | Data Inspection
Derivatives: Market Participation | Yield Analytics

## Tool identity (g3)
- Max 4 words
- NO protocol names, chain names, token names — ever
- End with one structural suffix:
  Series / Snapshot / Leaderboard / Tracker / Screener / Lookup / Feed / Monitor

## Variant (g4)
What metric this specific query measures. Max 4 words. No protocol names.

## Provenance (g5)
One sentence, max 10 words. May name protocols/chains/tokens. Never shown to users.

## Description
Generic one-sentence summary starting with a verb. NO protocol/chain/token names.
This gets embedded for semantic search — keep it abstract and reusable.

## Registry match
Check the registry below. Match on g3 first, then g4.

- g3 + g4 match → match: true, reuse tool_id
- g3 match, g4 differs → match: false, null tool_id (new variant row)
- No g3 match → match: false, null tool_id (new tool)

## Input rules
Extract only user-facing runtime parameters:
- Addresses, token symbols, protocol names, chain names
- Dates, lookback windows, time granularity
- Numeric thresholds, top-N limits

Never extract: table names, schema names, CTE names, SQL internals.
Dune {{param}} → always extract as required input.
Hardcoded value → extract with that value as default, required: false.
No value → default: null, required: true.
Max 4 inputs. Drop least meaningful if more than 4.

## Current registry (g3 + g4 + description only — check before creating)

{registry_json}

## Query to classify

Title: {title}

SQL:
{sql}

Return ONLY valid JSON. No markdown, no explanation, no preamble.
"""


def build_stage2_prompt(
    title: str,
    sql: str,
    registry_json: str,
    domains: list[dict],
) -> str:
    domain_list = "\n".join([
        f"  - {d['name']}: {d['description']}"
        for d in domains
    ])

    # slim registry — g3 + g4 + description only, no inputs, no source count
    import json
    try:
        tools = json.loads(registry_json)
        slim = [
            {"g3": t["g3"], "g4": t["g4"], "description": t["description"]}
            for t in tools
        ]
        slim_registry = json.dumps(slim, indent=2)
    except Exception:
        slim_registry = registry_json

    return STAGE2_PROMPT.format(
        domains=domain_list,
        registry_json=slim_registry,
        title=title,
        sql=sql,
    )