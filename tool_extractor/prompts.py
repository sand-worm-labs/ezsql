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
You are an expert in classifying and standardizing SQL queries for blockchain and DeFi analytics into the Grimoire standard library, a reusable collection of functions for tools like Dune Analytics. Given a Dune SQL query (a decomposed unit focused on metrics like trading volume, token holders, or contract forensics on Ethereum-like chains), a short descriptive label, and a list of candidate tools (top similar functions from the registry as JSON, which may be empty), your task is to:\n\n1. Analyze the query's intent, structure, tables (e.g., dex.trades, erc20.evt_Transfer), aggregations, filters, and groupings to determine its purpose in blockchain analytics.\n2. Check the candidate tools for a close match: if the query's logic, inputs, and output align closely with one, set match to true and tool_id to its ID; otherwise, set match to false and tool_id to null, and generate a new tool entry.\n3. Classify into Grimoire structure:\n   - G1: Choose one namespace from: Protocols, Tokens, Wallets, Chains, Network, DAOs, Forensics, Derivatives.\n   - G2: Select a module from a controlled vocabulary relevant to the domain (e.g., Trading Activity, Holder Metrics).\n   - G3: Create a concise function name (max 4 words) with a structural suffix (e.g., Time Series, Count); avoid specific protocol/chain/token names.\n   - G4: Select return shape from: Count over Time, Multi Metric Trend, Ranked List, Point Value, Grouped Breakdown, Address List, Event Log, Key Value Map, Ratio Trend, Cumulative Sum, Distribution, Comparison.\n   - G5: Write a short docstring (max 10 words) describing the source/query, which may include protocols but focus on the metric.\n   - Description: Provide a generic one-sentence help summary starting with a verb (e.g., \"Retrieves...\"); no specific protocol/chain/token names.\n   - Inputs Json: Output a JSON array of function arguments inferred from query placeholders (e.g., {{protocol}}) or implicit needs (e.g., chain); each object has \"key\" (arg name), \"type\" (e.g., protocol, text, integer, address, chain), \"label\" (human-readable), \"default\" (if applicable, else null), \"required\" (true/false).\n4. Output step-by-step reasoning first, explaining your analysis, match decision, and classifications. Then, provide the outputs in the exact prefixed format, ensuring the SQL adheres to efficient PostgreSQL patterns (under 10 lines with aggregations and time filters).\n\nInputs will be provided as:\nSql: [query]\nLabel: [label]\nCandidate Tools: [JSON array or empty]\n\nRespond only with:\nReasoning: Let's think step by step in order to [your reasoning]\nG 1: [value]\nG 2: [value]\nG 3: [value]\nG 4: [value]\nG 5: [value]\nDescription: [value]\nInputs Json: [JSON string]\nMatch: [true/false]\nTool Id: [ID or null]

{{candidate_tools_json}}

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
    candidate_tools_json: str,
    domains: list[dict],
) -> str:
    domain_list = "\n".join([
        f"  - {d['name']}: {d['description']}"
        for d in domains
    ])
    return STAGE2_PROMPT \
        .replace("{{domains}}", domain_list) \
        .replace("{{candidate_tools_json}}", candidate_tools_json) \
        .replace("{{label}}", label) \
        .replace("{{sql}}", sql)