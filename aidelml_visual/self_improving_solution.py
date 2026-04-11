import pandas as pd
import numpy as np
import re
import os
import math
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering

# Load data
train_df = pd.read_csv("./aidelml_visual/input/train.csv")
test_df = pd.read_parquet("./aidelml_visual/input/test.parquet")    

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
os.makedirs("./aidelml_visual/working", exist_ok=True)
# ============================================================
# PHASE 1: MINE THE CORPUS — learn what's actually in the data
# ============================================================

def extract_tables_from_sql(sql):
    """Extract table references from SQL."""
    if not sql or str(sql) == "nan":
        return []
    return re.findall(r"(?:from|join)\s+([a-zA-Z0-9_.]+)", str(sql).lower())


def get_table_prefix(table_name):
    """Get the schema/namespace prefix from a table name."""
    parts = table_name.split(".")
    return parts[0]


print("Mining table namespaces from corpus...")
all_tables = []
all_prefixes = []

# Mine from BOTH train and test
for df_name, df in [("train", train_df), ("test", test_df)]:
    for sql in df["query_sql"].dropna():
        tables = extract_tables_from_sql(sql)
        all_tables.extend(tables)
        for t in tables:
            all_prefixes.append(get_table_prefix(t))

prefix_counts = Counter(all_prefixes)
table_counts = Counter(all_tables)

print(f"Unique table prefixes: {len(prefix_counts)}")
print(f"Top 30 prefixes:")
for name, c in prefix_counts.most_common(30):
    print(f"  {name}: {c}")

# ============================================================
# PHASE 2: AUTO-BUILD DOMAIN MAP from corpus frequencies
# ============================================================

# Seed mappings for known Dune namespaces (these are stable/canonical)
SEED_DOMAIN_MAP = {
    # Spellbook sector spells
    "dex": "dex", "dex_aggregator": "dex",
    "nft": "nft", "nft_ethereum": "nft", "nft_polygon": "nft",
    "lending": "lending",
    "stablecoin": "stablecoins", "stablecoins": "stablecoins",
    "prices": "prices",
    "tokens": "transfers", "erc20": "transfers",
    "erc721": "nft", "erc1155": "nft",
    "balances": "transfers",
    "labels": "identity",
    "bridge": "bridge",
    # Raw chain tables
    "ethereum": "infra", "optimism": "infra", "arbitrum": "infra",
    "polygon": "infra", "bnb": "infra", "base": "infra",
    "avalanche": "infra", "gnosis": "infra", "fantom": "infra",
    "solana": "infra", "zksync": "infra", "scroll": "infra",
    "linea": "infra", "blast": "infra", "celo": "infra",
    "mantle": "infra", "zora": "infra",
    # Chain-specific raw data patterns
    "ethereum_ethereum": "infra", "polygon_polygon": "infra",
    # Protocol-specific decoded tables
    "uniswap": "dex", "uniswap_v2": "dex", "uniswap_v3": "dex",
    "sushiswap": "dex", "curve": "dex", "balancer": "dex",
    "pancakeswap": "dex", "trader_joe": "dex", "camelot": "dex",
    "velodrome": "dex", "aerodrome": "dex",
    "aave": "lending", "aave_v2": "lending", "aave_v3": "lending",
    "compound": "lending", "compound_v2": "lending", "compound_v3": "lending",
    "maker": "lending", "morpho": "lending", "euler": "lending",
    "spark": "lending", "radiant": "lending",
    "lido": "staking", "rocketpool": "staking", "eigenlayer": "staking",
    "opensea": "nft", "blur": "nft", "seaport": "nft",
    "looksrare": "nft", "x2y2": "nft", "sudoswap": "nft",
    "gmx": "derivatives", "dydx": "derivatives", "synthetix": "derivatives",
    "perpetual": "derivatives", "kwenta": "derivatives",
    "layerzero": "bridge", "stargate": "bridge", "hop": "bridge",
    "across": "bridge", "celer": "bridge", "synapse": "bridge",
    "wormhole": "bridge",
    "chainlink": "oracle", "pyth": "oracle",
    "ens": "identity", "lens": "identity",
    "safe": "governance", "gnosis_safe": "governance",
    "governor": "governance", "snapshot": "governance",
    "flashbots": "mev",
    # Common sub-table patterns
    "evt_transfer": "transfers", "evt_swap": "dex",
    "evt_mint": "transfers", "evt_burn": "transfers",
    "evt_approval": "transfers",
    "evt_borrow": "lending", "evt_repay": "lending",
    "evt_liquidation": "lending", "evt_deposit": "lending",
}

def auto_classify_prefix(prefix):
    """Try to classify an unknown prefix by substring matching against seeds."""
    prefix_lower = prefix.lower()

    # Direct match
    if prefix_lower in SEED_DOMAIN_MAP:
        return SEED_DOMAIN_MAP[prefix_lower]

    # Substring match against known protocols
    for seed, domain in SEED_DOMAIN_MAP.items():
        if seed in prefix_lower or prefix_lower in seed:
            return domain

    # Check for chain name patterns (decoded tables: protocol_chain.contract_evt_Event)
    chain_names = [
        "ethereum", "polygon", "arbitrum", "optimism", "bnb", "base",
        "avalanche", "fantom", "gnosis", "solana", "zksync", "scroll",
        "linea", "blast", "celo", "mantle", "zora", "abstract",
    ]
    for chain in chain_names:
        if chain in prefix_lower:
            # It's a decoded protocol table on a specific chain
            # Try to identify protocol from remaining parts
            cleaned = prefix_lower.replace(chain, "").strip("_")
            if cleaned in SEED_DOMAIN_MAP:
                return SEED_DOMAIN_MAP[cleaned]

    return None  # Unknown


# Build the full domain map from corpus
print("\nBuilding domain map from corpus...")
DOMAIN_MAP = {}
unknown_prefixes = []

for prefix, count in prefix_counts.most_common(500):
    domain = auto_classify_prefix(prefix)
    if domain:
        DOMAIN_MAP[prefix] = domain
    else:
        unknown_prefixes.append((prefix, count))

print(f"Mapped {len(DOMAIN_MAP)} prefixes to domains")
print(f"Unknown prefixes ({len(unknown_prefixes)}):")
for p, c in unknown_prefixes[:20]:
    print(f"  {p}: {c}")

# For unknowns, try to infer from the FULL table names that use this prefix
for prefix, count in unknown_prefixes:
    # Look at actual table names with this prefix
    related_tables = [t for t in table_counts if get_table_prefix(t) == prefix]
    table_text = " ".join(related_tables).lower()

    # Heuristic domain detection from table name content
    if any(x in table_text for x in ["swap", "pair", "pool", "liquidity", "amm", "router"]):
        DOMAIN_MAP[prefix] = "dex"
    elif any(x in table_text for x in ["borrow", "lend", "collateral", "liquidat", "repay"]):
        DOMAIN_MAP[prefix] = "lending"
    elif any(x in table_text for x in ["nft", "erc721", "erc1155", "token_id", "collection"]):
        DOMAIN_MAP[prefix] = "nft"
    elif any(x in table_text for x in ["stake", "deposit", "validator", "delegate", "restake"]):
        DOMAIN_MAP[prefix] = "staking"
    elif any(x in table_text for x in ["bridge", "relay", "message", "cross"]):
        DOMAIN_MAP[prefix] = "bridge"
    elif any(x in table_text for x in ["vote", "proposal", "governor", "dao"]):
        DOMAIN_MAP[prefix] = "governance"
    elif any(x in table_text for x in ["price", "oracle", "feed"]):
        DOMAIN_MAP[prefix] = "prices"
    elif any(x in table_text for x in ["transfer", "balance", "erc20"]):
        DOMAIN_MAP[prefix] = "transfers"
    else:
        DOMAIN_MAP[prefix] = "general"  # fallback

print(f"Final domain map: {len(DOMAIN_MAP)} entries")

# Domain distribution
domain_dist = Counter()
for prefix, count in prefix_counts.items():
    domain = DOMAIN_MAP.get(prefix, "general")
    domain_dist[domain] += count
print("\nDomain distribution:")
for d, c in domain_dist.most_common():
    print(f"  {d}: {c}")


# ============================================================
# PHASE 3: RULE-BASED g5 CLASSIFIER
# ============================================================

def classify_g5(sql):
    """Classify structural archetype from SQL structure."""
    if not sql or str(sql) == "nan":
        return "time_series"
    sql_lower = str(sql).lower()

    has_order_desc = bool(re.search(r"order\s+by\s+.*\bdesc\b", sql_lower))
    has_limit = bool(re.search(r"\blimit\b\s+\d+", sql_lower))
    has_group_by = "group by" in sql_lower
    has_date_trunc = "date_trunc" in sql_lower or "date_format" in sql_lower
    has_date_group = bool(re.search(
        r"group\s+by.*(?:date|day|week|month|hour|time|block_time|minute|year|quarter)",
        sql_lower, re.DOTALL
    ))
    has_time_select = bool(re.search(
        r"select.*(?:date_trunc|date_format|to_date|from_unixtime)",
        sql_lower, re.DOTALL
    ))
    has_address_filter = bool(re.search(
        r"where.*(?:address|wallet|owner)\s*(?:=|in)\s*(?:0x|'0x|\{\{)",
        sql_lower, re.DOTALL
    ))
    has_order_time = bool(re.search(
        r"order\s+by\s+.*(?:time|block_time|block_number|evt_block_time|timestamp)",
        sql_lower
    ))

    # Leaderboard: ranked list
    if has_order_desc and has_limit and not has_date_group and not has_date_trunc:
        return "leaderboard"

    # Lookup: specific entity query
    if has_address_filter and not has_group_by:
        return "lookup"

    # Time series: aggregated over time
    if has_date_group or has_date_trunc or has_time_select:
        return "time_series"

    # Snapshot: aggregated but not over time
    if has_group_by and not has_date_group:
        return "snapshot"

    # Series: raw events ordered by time
    if has_order_time and not has_group_by:
        return "series"

    # Fallback
    if not has_group_by and not has_order_desc:
        return "series"

    return "time_series"


# ============================================================
# PHASE 4: DOMAIN DETECTION PER QUERY
# ============================================================

def detect_query_domain(sql, name="", tags=""):
    """Detect primary domain of a query."""
    domains = []

    # From table names
    tables = extract_tables_from_sql(sql)
    for t in tables:
        prefix = get_table_prefix(t)
        if prefix in DOMAIN_MAP:
            domains.append(DOMAIN_MAP[prefix])
        # Also check full table name parts
        for part in t.replace(".", "_").split("_"):
            if part in SEED_DOMAIN_MAP:
                domains.append(SEED_DOMAIN_MAP[part])

    # From name/tags
    text = f"{name} {tags}".lower()
    keyword_to_domain = {
        "swap": "dex", "dex": "dex", "liquidity": "dex", "amm": "dex", "pool": "dex",
        "trade": "dex", "uniswap": "dex", "sushi": "dex", "curve": "dex",
        "lend": "lending", "borrow": "lending", "aave": "lending", "compound": "lending",
        "liquidat": "lending", "collateral": "lending",
        "nft": "nft", "opensea": "nft", "blur": "nft", "collection": "nft",
        "mint": "nft", "erc721": "nft",
        "stake": "staking", "validator": "staking", "lido": "staking",
        "eigenlayer": "staking", "restake": "staking",
        "bridge": "bridge", "cross-chain": "bridge", "layerzero": "bridge",
        "gas": "infra", "block": "infra", "transaction": "infra",
        "transfer": "transfers", "erc20": "transfers", "balance": "transfers",
        "stablecoin": "stablecoins", "usdt": "stablecoins", "usdc": "stablecoins",
        "peg": "stablecoins",
        "price": "prices", "oracle": "oracle",
        "mev": "mev", "sandwich": "mev", "arbitrage": "mev", "flashbot": "mev",
        "governance": "governance", "vote": "governance", "dao": "governance",
        "proposal": "governance",
        "perp": "derivatives", "futures": "derivatives", "options": "derivatives",
        "gmx": "derivatives", "dydx": "derivatives",
    }
    for kw, domain in keyword_to_domain.items():
        if kw in text:
            domains.append(domain)

    if not domains:
        return "general"
    return Counter(domains).most_common(1)[0][0]


# ============================================================
# PHASE 5: BUILD TOOLBOX — data-driven tool generation
# ============================================================

# Core tools per domain x archetype
# Instead of 500 hand-crafted tools, we use ~50 that cover the real clusters

DOMAIN_TO_G1G2 = {
    "dex": ("defi", "dex"),
    "lending": ("defi", "lending"),
    "staking": ("defi", "staking"),
    "derivatives": ("defi", "derivatives"),
    "nft": ("nft", "marketplace"),
    "infra": ("infrastructure", "contracts"),
    "transfers": ("transfers", "flow"),
    "bridge": ("infrastructure", "bridge"),
    "governance": ("governance", "voting"),
    "stablecoins": ("stablecoins", "peg"),
    "mev": ("infrastructure", "mev"),
    "prices": ("market", "price"),
    "oracle": ("infrastructure", "oracle"),
    "identity": ("identity", "labels"),
    "general": ("activity", "general"),
}

# g5 -> list of (g3_suffix, g4)
ARCHETYPE_TEMPLATES = {
    "time_series": [
        ("{domain}_volume_timeseries", "date|volume|count"),
        ("{domain}_activity_timeseries", "date|active_count"),
        ("{domain}_value_timeseries", "date|value_usd"),
    ],
    "leaderboard": [
        ("top_{domain}_leaderboard", "address|metric|count"),
    ],
    "snapshot": [
        ("{domain}_distribution_snapshot", "bucket|count"),
        ("{domain}_breakdown_snapshot", "category|amount|pct"),
    ],
    "series": [
        ("{domain}_event_series", "time|entity|action|amount"),
    ],
    "lookup": [
        ("{domain}_entity_lookup", "address|attribute|value"),
    ],
}

# Generate toolbox
TOOL_LIST = []  # (g3, g4, g5, domain, text_repr)

# For each domain that actually appears in the data, create tools
active_domains = [d for d, c in domain_dist.most_common() if c > 10]
print(f"\nActive domains: {active_domains}")

for domain in active_domains:
    for g5, templates in ARCHETYPE_TEMPLATES.items():
        for g3_template, g4 in templates:
            g3 = g3_template.format(domain=domain)
            # Build text representation from domain keywords
            domain_keywords = [k for k, v in SEED_DOMAIN_MAP.items() if v == domain][:15]
            text = f"{domain} {g3} {g4} {' '.join(domain_keywords)}"
            TOOL_LIST.append((g3, g4, g5, domain, text))

# ============================================================
# Feature extraction function (needed before toolbox generation)
# ============================================================

def get_query_features(row):
    """Extract features for matching."""
    parts = []
    sql = str(row.get("query_sql", "") or "")
    name = str(row.get("name", "") or "")
    desc = str(row.get("description", "") or "")
    tags = str(row.get("tags", "") or "")

    parts.append(name)
    parts.append(name)
    parts.append(desc)
    parts.append(tags)

    if sql and sql != "nan":
        sql_lower = sql.lower()
        tables = re.findall(r"(?:from|join)\s+([a-zA-Z0-9_.]+)", sql_lower)
        for t in tables:
            parts.extend([t] * 3)
            for p in t.replace(".", "_").split("_"):
                if len(p) > 2:
                    parts.append(p)

        # Domain keywords
        domain_words = re.findall(
            r"\b(swap|transfer|liquidity|price|volume|trade|user|token|protocol|"
            r"revenue|tvl|lending|staking|nft|bridge|governance|gas|balance|"
            r"yield|perpetual|perp|whale|flow|distribution|market|supply|"
            r"transaction|block|contract|event|borrow|collateral|liquidat|"
            r"validator|delegate|holder|wallet|fee|reward|stake|vote|"
            r"pool|dex|amm|mev|sandwich|arbitrage|stablecoin|peg|"
            r"mint|burn|airdrop|claim|oracle|blob|deposit|withdrawal|"
            r"profit|loss|pnl|erc20|erc721)\b",
            sql_lower,
        )
        parts.extend(domain_words)

    return " ".join(parts)

# Also add training examples as explicit tools (they're gold labels)
# Give them RICH text so they win similarity matches
for _, row in train_df.iterrows():
    sql = str(row.get("query_sql", "") or "")
    name = str(row.get("name", "") or "")
    tags = str(row.get("tags", "") or "")
    desc = str(row.get("description", "") or "")
    domain = detect_query_domain(sql, name, tags)
    # Build rich text: g3 label (weighted) + full query features
    query_features = get_query_features(row)
    g3_words = row["g3"].replace("_", " ")
    text = f"{g3_words} {g3_words} {g3_words} {row['g4']} {domain} {query_features}"
    TOOL_LIST.append((row["g3"], row["g4"], row["g5"], domain, text))

# Deduplicate by g3
seen_g3 = set()
TOOLS = []
for t in TOOL_LIST:
    if t[0] not in seen_g3:
        seen_g3.add(t[0])
        TOOLS.append(t)

toolbox_g3 = [t[0] for t in TOOLS]
toolbox_g4 = [t[1] for t in TOOLS]
toolbox_g5 = [t[2] for t in TOOLS]
toolbox_domain = [t[3] for t in TOOLS]
toolbox_texts = [t[4] for t in TOOLS]

TOOLBOX_SIZE = len(TOOLS)
print(f"\nToolbox size: {TOOLBOX_SIZE}")
print(f"log2(toolbox): {math.log2(TOOLBOX_SIZE):.2f}")


# ============================================================
# PHASE 6: VECTORIZE
# ============================================================

print("\nVectorizing...")
train_texts = [get_query_features(row) for _, row in train_df.iterrows()]

all_texts = toolbox_texts + train_texts
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
)
vectorizer.fit(all_texts)
toolbox_vectors = vectorizer.transform(toolbox_texts)


# ============================================================
# PHASE 7: PREDICT with domain-boosted two-stage matching
# ============================================================

def predict_single(row):
    """Predict g1-g5 for a single query."""
    sql = str(row.get("query_sql", "") or "")
    name = str(row.get("name", "") or "")
    tags = str(row.get("tags", "") or "")

    g5 = classify_g5(sql)
    domain = detect_query_domain(sql, name, tags)

    query_text = get_query_features(row)
    query_vec = vectorizer.transform([query_text])

    # Match against FULL toolbox (no hard g5 filter)
    sims = cosine_similarity(query_vec, toolbox_vectors)[0]

    # Soft boosts
    for j in range(len(toolbox_g5)):
        if toolbox_domain[j] == domain:
            sims[j] *= 1.5
        if toolbox_g5[j] == g5:
            sims[j] *= 1.3  # g5 match bonus, not hard filter

    best_idx = np.argmax(sims)

    g3 = toolbox_g3[best_idx]
    g4 = toolbox_g4[best_idx]
    g5_out = toolbox_g5[best_idx]
    d = toolbox_domain[best_idx]
    g1, g2 = DOMAIN_TO_G1G2.get(d, ("activity", "general"))

    return g1, g2, g3, g4, g5_out


# ============================================================
# PHASE 8: EVALUATE ON TRAIN
# ============================================================

print("\n=== Training set evaluation ===")
correct = 0
for i, (_, row) in enumerate(train_df.iterrows()):
    g1, g2, g3, g4, g5 = predict_single(row)
    if g3 == train_df.iloc[i]["g3"] and g4 == train_df.iloc[i]["g4"]:
        correct += 1
    elif i < 15:
        print(f"  MISS: '{row.get('name','')}' -> pred={g3} actual={train_df.iloc[i]['g3']}")

train_cov = correct / len(train_df)
eff = train_cov / math.log2(TOOLBOX_SIZE) if TOOLBOX_SIZE > 1 else 0
print(f"Coverage: {correct}/{len(train_df)} = {train_cov:.4f}")
print(f"Efficiency: {eff:.4f}")


# ============================================================
# PHASE 9: OPTIMIZE TOOLBOX SIZE
# ============================================================
# After seeing which tools are actually used, prune unused ones

print("\n=== Optimizing toolbox size ===")

# Quick scan: which tools get matched by training data?
used_tools = set()
for _, row in train_df.iterrows():
    _, _, g3, g4, g5 = predict_single(row)
    used_tools.add(g3)

# Check a sample of test data too
sample_size = min(5000, len(test_df))
test_sample = test_df.sample(n=sample_size, random_state=42)
for _, row in test_sample.iterrows():
    _, _, g3, g4, g5 = predict_single(row)
    used_tools.add(g3)

print(f"Tools actually used: {len(used_tools)} out of {TOOLBOX_SIZE}")

# Prune to only used tools + a small buffer
pruned_indices = [i for i, g3 in enumerate(toolbox_g3) if g3 in used_tools]

if len(pruned_indices) < TOOLBOX_SIZE:
    toolbox_g3_p = [toolbox_g3[i] for i in pruned_indices]
    toolbox_g4_p = [toolbox_g4[i] for i in pruned_indices]
    toolbox_g5_p = [toolbox_g5[i] for i in pruned_indices]
    toolbox_domain_p = [toolbox_domain[i] for i in pruned_indices]
    toolbox_vectors_p = toolbox_vectors[pruned_indices]

    new_size = len(pruned_indices)
    new_eff = train_cov / math.log2(new_size) if new_size > 1 else 0
    print(f"Pruned toolbox: {new_size} tools")
    print(f"New efficiency estimate: {new_eff:.4f} (was {eff:.4f})")

    # If pruning helps, use pruned versions
    if new_eff >= eff:
        print("Pruning improves efficiency, using pruned toolbox.")
        toolbox_g3 = toolbox_g3_p
        toolbox_g4 = toolbox_g4_p
        toolbox_g5 = toolbox_g5_p
        toolbox_domain = toolbox_domain_p
        toolbox_vectors = toolbox_vectors_p
        TOOLBOX_SIZE = new_size
        eff = new_eff


# ============================================================
# PHASE 10: FULL TEST PREDICTION
# ============================================================

print(f"\nPredicting {len(test_df)} test queries...")

BATCH_SIZE = 5000
results = {"g1": [], "g2": [], "g3": [], "g4": [], "g5": []}

# Precompute outside loop
toolbox_domain_arr = np.array(toolbox_domain)
toolbox_g5_arr = np.array(toolbox_g5)

for start in range(0, len(test_df), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(test_df))
    batch = test_df.iloc[start:end]

    batch_texts = [get_query_features(row) for _, row in batch.iterrows()]
    batch_vecs = vectorizer.transform(batch_texts)
    batch_g5s = [classify_g5(str(row.get("query_sql", "") or "")) for _, row in batch.iterrows()]
    batch_domains = [
        detect_query_domain(
            str(row.get("query_sql", "") or ""),
            str(row.get("name", "") or ""),
            str(row.get("tags", "") or ""),
        )
        for _, row in batch.iterrows()
    ]

    # Match against full toolbox
    sims = cosine_similarity(batch_vecs, toolbox_vectors)

    # Vectorized soft boosts
    for i in range(len(batch)):
        domain_mask = (toolbox_domain_arr == batch_domains[i])
        g5_mask = (toolbox_g5_arr == batch_g5s[i])
        sims[i, domain_mask] *= 1.5
        sims[i, g5_mask] *= 1.3

    best_indices = np.argmax(sims, axis=1)
    for i, best_idx in enumerate(best_indices):
        g3 = toolbox_g3[best_idx]
        g4 = toolbox_g4[best_idx]
        g5_f = toolbox_g5[best_idx]
        d = toolbox_domain[best_idx]
        g1, g2 = DOMAIN_TO_G1G2.get(d, ("activity", "general"))
        results["g1"].append(g1)
        results["g2"].append(g2)
        results["g3"].append(g3)
        results["g4"].append(g4)
        results["g5"].append(g5_f)

    if start % 100000 == 0:
        print(f"  {start}/{len(test_df)}")

# ============================================================
# PHASE 11: SAVE
# ============================================================

submission = pd.DataFrame({
    "query_id": test_df["query_id"],
    "g1": results["g1"],
    "g2": results["g2"],
    "g3": results["g3"],
    "g4": results["g4"],
    "g5": results["g5"],
})

submission.to_csv("./aidelml_visual/working/submission.csv", index=False)

actual_toolbox = len(set(zip(results["g3"], results["g4"])))
final_eff = train_cov / math.log2(actual_toolbox) if actual_toolbox > 1 else 0

print(f"\n=== FINAL METRICS ===")
print(f"Toolbox size (used): {actual_toolbox}")
print(f"Train coverage: {train_cov:.4f}")
print(f"Efficiency: {final_eff:.4f}")
print(f"Submission shape: {submission.shape}")

print(f"\nTop 20 predicted tools:")
pair_counts = Counter(zip(results["g3"], results["g5"]))
for pair, count in pair_counts.most_common(20):
    print(f"  {pair[0]} ({pair[1]}): {count}")

print(f"\ng5 distribution:")
for g5, count in Counter(results["g5"]).most_common():
    print(f"  {g5}: {count} ({count/len(results['g5'])*100:.1f}%)")

print(f"\nDomain distribution:")
domain_counts = Counter()
for g3 in results["g3"]:
    for t in TOOLS:
        if t[0] == g3:
            domain_counts[t[3]] += 1
            break
for d, c in domain_counts.most_common():
    print(f"  {d}: {c} ({c/len(results['g3'])*100:.1f}%)")

print(f"\nValidation Metric (Efficiency): {final_eff:.6f}")