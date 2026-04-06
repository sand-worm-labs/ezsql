import json
import os
import dspy

VALID_SHAPES = {
    "Count over Time",
    "Multi Metric Trend",
    "Ranked List",
    "Point Value",
    "Grouped Breakdown",
    "Address List",
    "Event Log",
    "Key Value Map",
    "Ratio Trend",
    "Cumulative Sum",
    "Distribution",
    "Comparison",
}

VALID_G1 = {
    "Protocols", "Tokens", "Wallets", "Chains",
    "Network", "DAOs", "Forensics", "Derivatives",
}

VALID_SUFFIXES = (
    "Series", "Snapshot", "Leaderboard", "Tracker",
    "Screener", "Lookup", "Feed", "Monitor",
)

# extend as needed
CHAIN_NAMES = {
    "ethereum", "solana", "polygon", "arbitrum", "optimism", "bsc",
    "avalanche", "fantom", "base", "starknet", "gnosis", "zksync",
    "scroll", "linea", "blast", "mantle", "celo", "lisk", "noble",
}

PROTOCOL_NAMES = {
    "uniswap", "aave", "curve", "dydx", "opensea", "pancakeswap",
    "sushiswap", "compound", "maker", "lido", "rocket", "balancer",
    "gmx", "synthetix", "yearn", "convex", "frax", "eigenlayer",
    "jupiter", "raydium", "orca", "marinade", "jito", "drift",
    "camelot", "thales", "clipper", "index_coop",
}

TOKEN_NAMES = {
    "eth", "btc", "usdc", "usdt", "dai", "weth", "wbtc", "steth",
    "arb", "op", "matic", "sol", "avax", "ftm", "bnb", "link",
}

def contaminated(text: str) -> bool:
    """Check if text contains protocol/chain/token names."""
    low = text.lower()
    return any(name in low for name in CHAIN_NAMES | PROTOCOL_NAMES | TOKEN_NAMES)


class ClassifyTool(dspy.Signature):
    """Classify a Dune SQL query into the grimoire standard library."""

    sql: str = dspy.InputField(desc="Decomposed SQL unit")
    label: str = dspy.InputField(desc="Short label from Stage 1")
    candidate_tools: str = dspy.InputField(desc="Top 10 similar functions from registry as JSON")

    g1: str = dspy.OutputField(desc="Namespace: Protocols|Tokens|Wallets|Chains|Network|DAOs|Forensics|Derivatives")
    g2: str = dspy.OutputField(desc="Module from controlled vocabulary")
    g3: str = dspy.OutputField(desc="Function name, max 4 words, structural suffix, no protocol/chain/token names")
    g4: str = dspy.OutputField(desc="Return shape: Count over Time|Multi Metric Trend|Ranked List|Point Value|Grouped Breakdown|Address List|Event Log|Key Value Map|Ratio Trend|Cumulative Sum|Distribution|Comparison")
    g5: str = dspy.OutputField(desc="Docstring: source description, may name protocols, max 10 words")
    description: str = dspy.OutputField(desc="help() — generic one-sentence summary starting with a verb, no protocol/chain/token names")
    inputs_json: str = dspy.OutputField(desc="JSON array of function arguments with key/type/label/default/required")
    match: bool = dspy.OutputField(desc="Whether this matches an existing function in candidates")
    tool_id: str = dspy.OutputField(desc="Matched tool_id or null")


# ─── MODULE ──────────────────────────────────────────────────────────────────

class GrimoireClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(ClassifyTool)

    def forward(self, sql, label, candidate_tools):
        return self.classify(sql=sql, label=label, candidate_tools=candidate_tools)



def tool_metric(example, prediction, trace=None) -> float:
    score = 0.0

    # ── g1 valid namespace ──
    if prediction.g1 in VALID_G1:
        score += 0.1

    # ── g3 clean function name ──
    if not contaminated(prediction.g3):
        score += 0.15
    else:
        score -= 0.3

    # g3 has valid suffix
    if any(prediction.g3.endswith(s) for s in VALID_SUFFIXES):
        score += 0.1

    # g3 max 4 words
    if len(prediction.g3.split()) <= 4:
        score += 0.05

    # ── g4 is a canonical return shape ──
    if prediction.g4 in VALID_SHAPES:
        score += 0.2
    else:
        score -= 0.2

    # ── g4 should NOT describe content ──
    if contaminated(prediction.g4):
        score -= 0.3

    # ── description differs from g5 ──
    if prediction.description.strip() != prediction.g5.strip():
        score += 0.15
    else:
        score -= 0.3

    # ── description has no protocol/chain/token ──
    if not contaminated(prediction.description):
        score += 0.1

    # ── description starts with a verb ──
    first_word = prediction.description.strip().split()[0].lower() if prediction.description.strip() else ""
    verb_starters = {
        "tracks", "returns", "counts", "ranks", "calculates", "computes",
        "identifies", "aggregates", "resolves", "segments", "lists",
        "detects", "monitors", "filters", "finds", "shows", "measures",
        "reports", "fetches", "retrieves", "maps", "summarizes",
    }
    if first_word in verb_starters:
        score += 0.05

    # ── inputs valid JSON, max 4, no oversized defaults ──
    try:
        inputs = json.loads(prediction.inputs_json)
        if isinstance(inputs, list) and len(inputs) <= 4:
            score += 0.1
            for inp in inputs:
                d = inp.get("default")
                if d and isinstance(d, str):
                    if len(d) > 100:
                        score -= 0.15
                    if "," in d:
                        score -= 0.15
    except Exception:
        score -= 0.2

    return max(0.0, min(1.0, score))


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train():
    from dspy_data import TRAINSET 

    lm = dspy.LM(
        model="openrouter/x-ai/grok-4-fast", 
        api_base="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
        max_tokens=2048,
    )

    dspy.configure(lm=lm)

    classifier = GrimoireClassifier()

    optimizer = dspy.MIPROv2(
        metric=tool_metric,
        num_threads=4,
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
    )

    optimized = optimizer.compile(
        classifier,
        trainset=TRAINSET.TRAINSET,
        requires_permission_to_run=False,
    )

    optimized.save("grimoire_classifier_v3.json")
    print("Saved: grimoire_classifier_v3.json")
    return optimized


if __name__ == "__main__":
    train()