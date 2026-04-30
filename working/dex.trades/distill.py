#!/usr/bin/env python3
"""
distill.py — Grimoire query classifier (no LLM)
Usage: python distill.py --files Query_2_.sql Query_8_.sql --out mapping.json
"""

import re, json, argparse
from collections import Counter

TOOLS = [
    { "id": "defi.dex_token_trade_flow",              "g3": "dex_token_trade_flow" },
    { "id": "defi.dex_volume_by_protocol_timeseries", "g3": "dex_volume_by_protocol_timeseries" },
    { "id": "defi.dex_unique_trader_count",           "g3": "dex_unique_trader_count" },
    { "id": "defi.dex_wallet_token_activity",         "g3": "dex_wallet_token_activity" },
    { "id": "defi.dex_token_price_from_swaps",        "g3": "dex_token_price_from_swaps" },
    { "id": "defi.dex_pair_volume_timeseries",        "g3": "dex_pair_volume_timeseries" },
    { "id": "defi.dex_new_vs_returning_users",        "g3": "dex_new_vs_returning_users" },
    { "id": "defi.dex_trade_size_distribution",       "g3": "dex_trade_size_distribution" },
    { "id": "defi.dex_protocol_active_users",         "g3": "dex_protocol_active_users" },
    { "id": "defi.dex_protocol_market_share",         "g3": "dex_protocol_market_share" },
    { "id": "defi.dex_large_trades",                  "g3": "dex_large_trades" },
    { "id": "defi.dex_volume_by_chain",               "g3": "dex_volume_by_chain" },
    { "id": "defi.dex_top_tokens_by_volume",          "g3": "dex_top_tokens_by_volume" },
    { "id": "defi.dex_top_wallets_by_volume",         "g3": "dex_top_wallets_by_volume" },
    { "id": "defi.dex_protocol_chain_breakdown",      "g3": "dex_protocol_chain_breakdown" },
]

INSERT_RE = re.compile(r"INSERT INTO \"Query\" \([^)]+\) VALUES \(", re.IGNORECASE)
TABLE_RE  = re.compile(r'\bfrom\s+([\w]+\.[\w"]+)', re.IGNORECASE)

# ── parse ────────────────────────────────────────────────────────────────────

def parse_file(path):
    with open(path, "r", errors="replace") as f:
        content = f.read()
    blocks, results = INSERT_RE.split(content)[1:], []
    for block in blocks:
        qid_m  = re.match(r"'(\d+)'", block)
        name_m = re.match(r"'\d+', '([^']*)'", block)
        sql_m  = re.search(r"'v[12][^']*'\s*,\s*'(.*?)'\s*,\s*'(?:False|True)'", block, re.DOTALL)
        if not sql_m:
            sql_m = re.search(r"'\w[^']*\[(?:deprecated|Dune SQL)\]'\s*,\s*'(.*?)'\s*,\s*'(?:False|True)'", block, re.DOTALL)
        if not qid_m: continue
        sql    = sql_m.group(1).lower() if sql_m else ""
        tables = [t.strip('"').lower() for t in TABLE_RE.findall(sql)]
        results.append({"query_id": qid_m.group(1), "name": name_m.group(1) if name_m else "", "sql": sql, "primary_table": tables[0] if tables else "unknown"})
    return results

def load_all(files):
    seen, rows = set(), []
    for path in files:
        print(f"  parsing {path}...")
        for row in parse_file(path):
            if row["query_id"] not in seen:
                seen.add(row["query_id"])
                rows.append(row)
    print(f"  total unique queries: {len(rows)}")
    return rows

# ── classify ─────────────────────────────────────────────────────────────────

def pick(g3):
    for t in TOOLS:
        if t["g3"] == g3: return t["id"]
    return "UNCLASSIFIED"

def classify(name, sql):
    nm = name.lower()
    s  = re.sub(r'/\*.*?\*/', '', sql.strip(), flags=re.DOTALL)
    s  = re.sub(r'try_cast\s*\(', 'cast(', s)

    has_sum           = bool(re.search(r'sum\s*\(', s))
    has_count         = bool(re.search(r'count\s*\(', s))
    has_cd            = bool(re.search(r'count\s*\(\s*distinct', s))
    has_avg           = bool(re.search(r'avg\s*\(', s))
    has_max           = bool(re.search(r'max\s*\(', s))
    has_date          = bool(re.search(r'date_trunc|block_date|block_time', s))
    has_group         = 'group by' in s
    has_limit         = 'limit' in s
    has_addr          = bool(re.search(r'token_(bought|sold)_address\s*(=|in)', s))
    has_inline_addr   = bool(re.search(r'0x[0-9a-f]{10,}', s))
    has_wallet_filter = bool(re.search(r'(taker|maker|tx_from|tx_to)\s*(=|in)\s*(0x|\')', s))
    has_price_div     = bool(re.search(r'amount_usd\s*/\s*nullif|usd_amount\s*/\s*nullif', s))
    has_blockchain    = 'blockchain' in s
    has_project       = 'project' in s
    has_direction     = "'buy'" in s or "'sell'" in s or 'buy' in nm or 'sell' in nm
    group_tail        = s[s.rfind('group by'):] if 'group by' in s else ''

    if (any(x in nm for x in ['price','peg','market cap','pnl']) or has_price_div) and not any(x in nm for x in ['volume','share','trader','user','count']):
        return pick('dex_token_price_from_swaps')
    if any(x in s for x in ['histogram','power(2','floor(log','width_bucket','percentile','ntile']):
        return pick('dex_trade_size_distribution')
    if any(x in nm for x in ['histogram','distribution','percentile','median','avg tx','avg trade','avg swap','average tx','heatmap']):
        return pick('dex_trade_size_distribution')
    if has_avg and any(x in nm for x in ['avg','average','mean']):
        return pick('dex_trade_size_distribution')
    if any(x in s for x in ["min(block_time)","first_trade","'new'","'returning'","first_swap","first_seen","lag(","lead("]) or any(x in nm for x in ['new vs','returning','cohort','retention','activation','new user','first time','new/old','new & old']):
        return pick('dex_new_vs_returning_users')
    if any(x in nm for x in ['whale','largest','large trade','>500k','notable','flash']):
        return pick('dex_large_trades')
    if re.search(r'amount_usd\s*>\s*\d{4,}', s) or re.search(r'usd_amount\s*>\s*\d{4,}', s):
        return pick('dex_large_trades')
    if s.strip().startswith('create') or (has_max and not has_sum and not has_cd):
        return pick('dex_wallet_token_activity')
    if not has_sum and not has_cd and not has_group and not has_count:
        return pick('dex_wallet_token_activity')
    if has_wallet_filter and not has_group:
        return pick('dex_wallet_token_activity')
    if re.search(r'select\s*\*\s*from\s*dex', s) and not has_sum:
        return pick('dex_wallet_token_activity')
    flow_n = any(x in nm for x in ['bought & sold','buy and sell','buys on','sold & bought','net buyer','net seller','buy/sell','buys & sells','inflow','outflow','bought on','sold on','buy side','sell side','accumul','buyer','seller'])
    if (flow_n or ((has_addr or has_inline_addr) and has_sum) or (has_direction and has_sum)) and not any(x in nm for x in ['pair','pool','price','peg','market share']):
        return pick('dex_token_trade_flow')
    if any(x in nm for x in ['pair','pool ','pools ','3pool','curve','usdc-weth','weth-usdc','eth-usdt','token pair','top pair','stable pair','lp ']) or ('token_pair' in s and has_sum):
        return pick('dex_pair_volume_timeseries')
    if any(x in nm for x in ['market share','dominance','dex share','dex shares','aggregator share','volume share','dex rank','hhi']) or (re.search(r'/\s*(select\s+sum|sum\s*\()', s) and has_project):
        return pick('dex_protocol_market_share')
    if any(x in nm for x in ['by chain','by blockchain','multichain','cross chain','chain breakdown','evm chains','all chains','blockchain volume']):
        return pick('dex_volume_by_chain')
    if has_blockchain and has_sum and 'blockchain' in group_tail and not has_project:
        return pick('dex_volume_by_chain')
    if bool(re.search(r"project\s*=\s*['']{1,2}[^'']+['']{1,2}", s)) and has_blockchain and has_sum and 'blockchain' in group_tail:
        return pick('dex_protocol_chain_breakdown')
    if any(x in nm for x in ['top token','top asset','most bought','most traded','popular token','top symbol','trending token']) or ('token_bought_symbol' in s and has_sum and has_limit and not has_cd):
        return pick('dex_top_tokens_by_volume')
    if any(x in nm for x in ['top wallet','top trader','top swapper','top address','leaderboard','most active wallet','smart money']):
        return pick('dex_top_wallets_by_volume')
    if has_count and 'taker' in s and has_limit and 'order by' in s and not has_cd:
        return pick('dex_top_wallets_by_volume')
    if any(x in nm for x in ['mau','monthly active','weekly active','daily active','dau','wau','active user','active address']):
        return pick('dex_protocol_active_users')
    if has_cd and 'amount_usd >=' in s:
        return pick('dex_protocol_active_users')
    if any(x in nm for x in ['unique trader','unique address','number of trader','traders last','distinct trader','# unique','# trader']):
        return pick('dex_unique_trader_count')
    if has_cd and any(x in s for x in ['taker','trader','sender','tx_from']):
        return pick('dex_unique_trader_count')
    if has_count and not has_sum and has_group:
        return pick('dex_unique_trader_count')
    if any(x in nm for x in ['volume','vol ','vol,','dex volume','trading volume','trade volume','swap volume','total volume','cumulative']):
        return pick('dex_volume_by_protocol_timeseries')
    if has_sum and has_date and has_group:
        return pick('dex_volume_by_protocol_timeseries')
    if has_sum:                  return pick('dex_volume_by_protocol_timeseries')
    if has_cd:                   return pick('dex_unique_trader_count')
    if has_count and has_group:  return pick('dex_unique_trader_count')
    if has_wallet_filter:        return pick('dex_wallet_token_activity')
    if has_inline_addr:          return pick('dex_token_trade_flow')
    return "UNCLASSIFIED"

# ── report ───────────────────────────────────────────────────────────────────

def print_report(counts, total):
    print(f"\n{'─'*65}")
    print(f"  {'TOOL':52s} {'COUNT':>6}  {'%':>5}")
    print(f"{'─'*65}")
    for tool, cnt in counts.most_common():
        print(f"  {tool:52s} {cnt:6d}  ({cnt/total*100:.1f}%)")
    print(f"{'─'*65}")
    print(f"  {'TOTAL':52s} {total:6d}")
    unc = counts.get("UNCLASSIFIED", 0)
    if unc:
        print(f"\n  {unc} unclassified — send to LLM for new tool discovery")
    print()

# ── main ─────────────────────────────────────────────────────────────────────

def run(files, out_path, show_unclassified):
    print("\n[1] loading...")
    queries = load_all(files)
    total   = len(queries)

    print("\n[2] classifying...")
    mapping = [{"query_id": q["query_id"], "name": q["name"], "tool": classify(q["name"], q["sql"])} for q in queries]

    counts = Counter(m["tool"] for m in mapping)
    print_report(counts, total)

    if show_unclassified:
        print("UNCLASSIFIED:")
        for m in mapping:
            if m["tool"] == "UNCLASSIFIED":
                print(f"  [{m['query_id']}] {m['name']}")

    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--out", default="mapping.json")
    parser.add_argument("--unclassified", action="store_true")
    args = parser.parse_args()
    run(args.files, args.out, args.unclassified)
