"""
Regenerate g3/g4/g5 labels using:
  g3 = slugified query name  + g5 suffix (if not already present)
  g4 = pipe-joined SELECT column aliases
  g5 = SQL lego output shape detection
"""

import pandas as pd
import re
import os
import math
from collections import Counter, defaultdict

os.makedirs("./taxonomy_weird_solution/working", exist_ok=True)

# ── LOAD ──────────────────────────────────────────────────────────────────────
taxonomy = pd.read_csv("./taxonomy_weird_solution/input/table_taxonomy_output.csv")
train    = pd.read_csv("./taxonomy_weird_solution/input/train.csv")
test     = pd.read_parquet("./taxonomy_weird_solution/input/test.parquet")

print(f"Train: {train.shape}  Test: {test.shape}")

# ── G5 FROM SQL LEGOS ─────────────────────────────────────────────────────────
def detect_g5(sql: str) -> str:
    """Detect output shape from SQL structure."""
    if not isinstance(sql, str):
        return "scalar"
    s = sql.lower()
    s = re.sub(r"--[^\n]*", " ", s)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)

    has_rank      = bool(re.search(r"\b(rank|row_number|dense_rank|ntile)\s*\(", s))
    has_limit     = bool(re.search(r"\blimit\b", s))
    has_order     = bool(re.search(r"\border\s+by\b", s))
    has_time_trunc= bool(re.search(r"\b(date_trunc|time_bucket)\b", s))
    has_group     = bool(re.search(r"\bgroup\s+by\b", s))
    has_agg       = bool(re.search(r"\b(sum|count|avg|max|min)\s*\(", s))
    has_window    = bool(re.search(r"\bover\s*\(", s))
    has_case      = bool(re.search(r"\bcase\b", s))
    has_filter_agg= bool(re.search(r"\bfilter\s*\(\s*where\b", s))
    is_star       = bool(re.match(r"\s*select\s+\*", s))

    if has_rank or (has_order and has_limit and has_agg):
        return "leaderboard"
    if has_time_trunc and has_group:
        return "time_series"
    if has_window and (has_agg):
        return "time_series"      # cumulative → still time_series family
    if (has_case or has_filter_agg) and has_group and has_agg:
        return "breakdown"
    if has_agg and not has_group:
        return "scalar"
    if has_agg and has_group:
        return "scalar"           # grouped non-time = still scalar family
    if is_star or not has_agg:
        return "raw"
    return "scalar"


# ── G4 FROM SELECT ALIASES ────────────────────────────────────────────────────
def extract_select_aliases(sql: str) -> list:
    """
    Parse top-level SELECT clause and extract column aliases.
    Falls back to function names when no alias present.
    """
    if not isinstance(sql, str):
        return []
    s = re.sub(r"--[^\n]*", " ", sql)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)

    # SELECT * → raw, no meaningful aliases
    if re.match(r"\s*select\s+\*", s, re.IGNORECASE):
        return []

    # Strip leading WITH ... AS (...) CTEs to get the final SELECT
    # Find the LAST top-level SELECT (outermost query result)
    # Simple approach: find SELECT...FROM at depth 0
    depth    = 0
    pos      = 0
    sel_start = None
    i = 0
    s_lower = s.lower()
    while i < len(s_lower):
        if s_lower[i] == '(':
            depth += 1
        elif s_lower[i] == ')':
            depth -= 1
        elif depth == 0 and s_lower[i:i+6] == 'select':
            sel_start = i + 6
        i += 1

    if sel_start is None:
        return []

    # Find FROM at depth 0 after sel_start
    depth = 0
    sel_end = len(s)
    for i in range(sel_start, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
        elif depth == 0 and s_lower[i:i+5] in ('from ', 'from\n', 'from\t'):
            sel_end = i
            break

    select_clause = s[sel_start:sel_end].strip()

    # Split by top-level commas
    cols  = []
    depth = 0
    cur   = []
    for ch in select_clause:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            cols.append(''.join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur:
        cols.append(''.join(cur).strip())

    aliases = []
    for col in cols:
        col = col.strip()
        if not col:
            continue

        # explicit AS alias (last token after AS)
        m = re.search(r'\bas\s+([a-z_][a-z0-9_]*)\s*$', col, re.IGNORECASE)
        if m:
            aliases.append(m.group(1).lower())
            continue

        # bare identifier (no parens, no spaces)
        if re.match(r'^[a-z_][a-z0-9_]*$', col, re.IGNORECASE):
            aliases.append(col.lower())
            continue

        # function with no alias → use function name as alias
        m_fn = re.match(r'([a-z_][a-z0-9_]*)\s*\(', col, re.IGNORECASE)
        if m_fn:
            aliases.append(m_fn.group(1).lower())
            continue

        # table.col with no alias → use col name
        m_tc = re.match(r'[a-z_][a-z0-9_]*\.([a-z_][a-z0-9_]*)\s*$', col, re.IGNORECASE)
        if m_tc:
            aliases.append(m_tc.group(1).lower())
            continue

        # last word as implicit alias
        m_last = re.search(r'\b([a-z_][a-z0-9_]*)\s*$', col, re.IGNORECASE)
        if m_last:
            word = m_last.group(1).lower()
            # skip SQL keywords as implicit aliases
            if word not in {'asc','desc','null','true','false','else','end','then'}:
                aliases.append(word)

    return aliases


# ── G3 FROM NAME ──────────────────────────────────────────────────────────────
# Protocol/token names to strip (leave domain semantics, remove branding)
_STRIP = {
    # protocols
    'uniswap','sushiswap','curve','balancer','aave','compound','morpho',
    'fraxlend','fraxfinance','velodrome','aerodrome','gmx','dydx','perp',
    # chains
    'ethereum','arbitrum','optimism','polygon','bnb','avalanche','base',
    'zksync','linea','fantom','gnosis','solana','bitcoin',
    # version tokens
    'v1','v2','v3','v4',
    # exchange names
    'opensea','blur','looksrare','x2y2','rarible','foundation',
    'cryptopunks','mayc','bayc','azuki',
    # project names
    'ftx','etherfi','aztec','layerzero','nouns','collab','voltz','link3',
    'fort','cvx','fold','frax','fxs','glpeth',
    # noise words that don't add meaning to g3
    'the','and','with','for','all','new','old','per',
    # time specifics that belong in description not g3
    'feb19','nov6','7d','24h','30d','90d','1m',
}
# Output type words already implied by g5 — normalise them in name
_OUTPUT_NORM = {
    'timeseries': 'timeseries', 'time series': 'timeseries',
    'time-series': 'timeseries',
    'leaderboard': 'leaderboard',
    'breakdown': 'breakdown',
    'distribution': 'breakdown',
    'scalar': 'scalar',
    'raw': 'raw',
}
# g5 → suffix appended to g3 if not already present
_G5_SUFFIX = {
    'time_series': 'timeseries',
    'leaderboard': 'leaderboard',
    'breakdown':   'breakdown',
    'scalar':      'scalar',
    'raw':         'raw',
}

def name_to_g3(name: str, g5: str) -> str:
    """Convert query name + output type → canonical g3 slug."""
    if not isinstance(name, str) or not name.strip():
        return f"generic_{g5}"

    s = name.lower()
    # Remove special chars, keep alphanumeric + spaces
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    # Remove numbers that are standalone (version/threshold noise)
    words = [w for w in s.split() if w not in _STRIP and len(w) > 1]

    # Normalise output-type words already in name
    normalised = []
    i = 0
    while i < len(words):
        pair = f"{words[i]} {words[i+1]}" if i+1 < len(words) else ""
        if pair in _OUTPUT_NORM:
            normalised.append(_OUTPUT_NORM[pair])
            i += 2
        elif words[i] in _OUTPUT_NORM:
            normalised.append(_OUTPUT_NORM[words[i]])
            i += 1
        else:
            normalised.append(words[i])
            i += 1

    slug = '_'.join(normalised).strip('_')
    slug = re.sub(r'_+', '_', slug)

    # Append g5 suffix if not already present
    suffix = _G5_SUFFIX.get(g5, '')
    if suffix and not slug.endswith(suffix):
        slug = f"{slug}_{suffix}"

    return slug


# ── TAXONOMY LOOKUP (for g1/g2) ───────────────────────────────────────────────
taxonomy["namespace"]  = taxonomy["namespace"].fillna("").str.lower().str.strip()
taxonomy["table_name"] = taxonomy["table_name"].str.lower().str.strip()

table_map = {}
ns_map    = defaultdict(list)
word_map  = defaultdict(list)

for _, row in taxonomy.iterrows():
    ns  = row["namespace"]
    tn  = row["table_name"]
    val = (row["g1"], row["g2"], row["g3"])
    table_map[tn] = val
    if ns:
        table_map[f"{ns}.{tn}"] = val
        ns_map[ns].append(val)
    for w in re.split(r"[_.\-]", tn):
        if len(w) > 3:
            word_map[w].append(val)

OVERRIDES = {
    "wyvernexchange_evt_ordersmatched":             ("nft",  "nft_marketplaces",  "orders_matched"),
    "opensea.wyvernexchange_evt_ordersmatched":     ("nft",  "nft_marketplaces",  "orders_matched"),
    "seaport_evt_orderfulfilled":                   ("nft",  "nft_marketplaces",  "order_fulfilled"),
    "cryptopunksmarket_evt_punkbought":             ("nft",  "nft_marketplaces",  "nft_bought"),
    "cryptopunks.cryptopunksmarket_evt_punkbought": ("nft",  "nft_marketplaces",  "nft_bought"),
    "erc20.evt_transfer":                           ("payments","token_contracts","token_transferred"),
    "erc20.erc20_evt_transfer":                     ("payments","token_contracts","token_transferred"),
    "optimism.transactions":                        ("infra","blockchain_data",   "transaction_recorded"),
    "arbitrum.transactions":                        ("infra","blockchain_data",   "transaction_recorded"),
    "polygon.transactions":                         ("infra","blockchain_data",   "transaction_recorded"),
    "bnb.logs":                                     ("infra","blockchain_data",   "event_logged"),
    "base.transactions":                            ("infra","blockchain_data",   "transaction_recorded"),
}
for k, v in OVERRIDES.items():
    table_map[k.lower()] = v

ns_best = {}
for ns, vals in ns_map.items():
    cnt = Counter(v[2] for v in vals)
    best = cnt.most_common(1)[0][0]
    for v in vals:
        if v[2] == best:
            ns_best[ns] = v
            break

_TABLE_RE = re.compile(
    r'(?:from|join)\s+'
    r'((?:[a-z_][a-z0-9_\-]*\.)?(?:"[^"]+"|`[^`]+`|[a-z_][a-z0-9_]*))',
    re.IGNORECASE
)
_KW_SKIP = {"where","select","group","order","having","limit","on","and","or",
            "not","in","as","by","left","right","inner","outer","cross","lateral"}

def extract_tables(sql):
    if not isinstance(sql, str): return []
    s = re.sub(r"--[^\n]*", " ", sql.lower())
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
    return [t.strip().strip('"').strip('`') for t in _TABLE_RE.findall(s)
            if t.strip().strip('"').strip('`') not in _KW_SKIP]

def lookup_table(t):
    t = t.lower().strip()
    if t in table_map: return table_map[t]
    parts = t.split(".")
    if len(parts) >= 2:
        ns, tn = parts[-2], parts[-1]
        if f"{ns}.{tn}" in table_map: return table_map[f"{ns}.{tn}"]
        if tn in table_map: return table_map[tn]
        if ns in ns_best: return ns_best[ns]
    else:
        tn = parts[0]
        if tn in table_map: return table_map[tn]
    words = [w for w in re.split(r"[_.\-]", t) if len(w) > 3]
    if words:
        scores, meta = Counter(), {}
        for w in words:
            for v in word_map.get(w, []):
                scores[v[2]] += 1
                if v[2] not in meta: meta[v[2]] = v
        if scores:
            best = scores.most_common(1)[0][0]
            return meta[best]
    return None

_KW_DOMAIN = [
    (["perpetual","perp_","futures"],    ("derivatives","perpetual_protocols","position_settled")),
    (["bridge","layerzero","wormhole"],  ("bridge","bridge_protocols","bridge_requested")),
    (["vote","governance","dao"],        ("governance","voting_systems","vote_cast")),
    (["stake","staking","validator"],    ("staking","staking_pools","stake_deposited")),
    (["nft","erc721","opensea","blur"],  ("nft","nft_marketplaces","orders_matched")),
    (["borrow","liquidat","collateral"], ("lending","money_markets","debt_origination")),
    (["lend","aave","compound"],         ("lending","money_markets","token_deposited")),
    (["swap","uniswap","amm"],           ("defi","dex_amm","asset_exchange")),
    (["liquidity","pool","tvl"],         ("defi","liquidity_pools","liquidity_added")),
    (["transfer","erc20"],               ("payments","token_contracts","token_transferred")),
    (["price","oracle"," usd "],         ("analytics","price_feeds","usd_price_queried")),
    (["transaction","gas","fee"],        ("infra","blockchain_data","transaction_recorded")),
    (["block","miner"],                  ("infra","blockchain_data","block_recorded")),
]

def get_domain(sql):
    tables = extract_tables(sql)
    for t in tables:
        r = lookup_table(t)
        if r: return r
    s = sql.lower() if isinstance(sql, str) else ""
    for kws, val in _KW_DOMAIN:
        if any(k in s for k in kws): return val
    return ("infra","blockchain_data","transaction_recorded")


# ── FULL CLASSIFY ─────────────────────────────────────────────────────────────
def classify(name: str, sql: str, tags: str = ""):
    g5      = detect_g5(sql)
    g3      = name_to_g3(name, g5)
    aliases = extract_select_aliases(sql)
    g4      = "|".join(aliases) if aliases else g5   # fallback to g5 shape
    g1, g2, _ = get_domain(sql)
    return g1, g2, g3, g4, g5


# ── VALIDATE ON TRAIN ─────────────────────────────────────────────────────────
print("\n=== Train validation ===")
g3_match = g4_match = g5_match = 0

for _, row in train.iterrows():
    g1, g2, g3, g4, g5 = classify(row["name"], row["query_sql"], row.get("tags",""))
    g3_ok = g3 == row["g3"]
    g4_ok = g4 == row["g4"]
    g5_ok = g5 == row["g5"]
    if g3_ok: g3_match += 1
    if g4_ok: g4_match += 1
    if g5_ok: g5_match += 1
    print(f"  g3{'✓' if g3_ok else '✗'} pred={g3:<55} actual={row['g3']}")
    print(f"  g4{'✓' if g4_ok else '✗'} pred={g4:<40} actual={row['g4']}")
    print(f"  g5{'✓' if g5_ok else '✗'} pred={g5:<15} actual={row['g5']}")
    print()

n = len(train)
print(f"g3 accuracy: {g3_match}/{n} = {g3_match/n:.3f}")
print(f"g4 accuracy: {g4_match}/{n} = {g4_match/n:.3f}")
print(f"g5 accuracy: {g5_match}/{n} = {g5_match/n:.3f}")


# ── REGENERATE TRAINING DATA ──────────────────────────────────────────────────
print("\nRegenerating training labels...")
regen_rows = []
for _, row in train.iterrows():
    g1, g2, g3, g4, g5 = classify(row["name"], row["query_sql"], row.get("tags",""))
    regen_rows.append({
        "query_id":    row["query_id"],
        "name":        row["name"],
        "description": row.get("description",""),
        "tags":        row.get("tags",""),
        "query_engine":row.get("query_engine",""),
        "query_sql":   row["query_sql"],
        "is_private":  row.get("is_private",""),
        "is_archived": row.get("is_archived",""),
        "owner":       row.get("owner",""),
        # original labels
        "g1_actual":   row["g1"], "g2_actual":  row["g2"],
        "g3_actual":   row["g3"], "g4_actual":  row["g4"], "g5_actual": row["g5"],
        # regenerated labels
        "g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5,
    })

regen_df = pd.DataFrame(regen_rows)
regen_df.to_csv("./taxonomy_weird_solution/working/train_regenerated.csv", index=False)
print(f"Saved train_regenerated.csv: {regen_df.shape}")


# ── PROCESS TEST ──────────────────────────────────────────────────────────────
print(f"\nProcessing {len(test):,} test queries...")
records = []
for _, row in test.iterrows():
    g1, g2, g3, g4, g5 = classify(
        row.get("name",""), row.get("query_sql",""), row.get("tags","")
    )
    records.append({
        "query_id":    row["query_id"],
        "name":        row.get("name",""),
        "description": row.get("description",""),
        "tags":        row.get("tags",""),
        "query_engine":row.get("query_engine",""),
        "query_sql":   row.get("query_sql",""),
        "is_private":  row.get("is_private",""),
        "is_archived": row.get("is_archived",""),
        "owner":       row.get("owner",""),
        "g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5,
    })

results = pd.DataFrame(records)

toolbox_size = results[["g3","g4"]].drop_duplicates().shape[0]
efficiency   = 1.0 / math.log2(max(toolbox_size, 2))
print(f"\nToolbox size: {toolbox_size}  Efficiency: {efficiency:.6f}")
print(f"\ng3 sample:\n{results['g3'].value_counts().head(15)}")
print(f"\ng5 dist:\n{results['g5'].value_counts()}")

# submission (labels only)
submission = results[["query_id","g1","g2","g3","g4","g5"]]
submission.to_csv("./taxonomy_weird_solution/working/submission.csv", index=False)

# enriched (everything)
results.to_parquet("./taxonomy_weird_solution/working/enriched.parquet", index=False)

print(f"\nSubmission: {submission.shape}")
print(f"Enriched  : {results.shape}")
print(submission.head(10).to_string())