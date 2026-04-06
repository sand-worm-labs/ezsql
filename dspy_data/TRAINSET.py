import dspy_data
import json

TRAINSET = [
    # ── 1. Volume + traders trended ──────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT date_trunc('day', block_time) AS day,
       SUM(amount_usd) AS volume,
       COUNT(DISTINCT taker) AS traders
FROM dex.trades
WHERE project = '{{protocol}}'
  AND block_time >= now() - interval '{{timeframe}}'
GROUP BY 1 ORDER BY 1
""",
        label="volume and traders over time",
        candidate_tools="[]",
        g1="Protocols",
        g2="Trading Activity",
        g3="Volume Traders Series",
        g4="Multi Metric Trend",
        g5="Clipper daily volume and traders last 7 days",
        description="Tracks trading volume and unique trader counts over time for a DEX protocol.",
        inputs_json=json.dumps([
            {"key": "protocol", "type": "protocol", "label": "Protocol", "default": "clipper", "required": False},
            {"key": "timeframe", "type": "text", "label": "Timeframe", "default": "7 days", "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 2. Top trading pairs ─────────────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT token_pair, SUM(amount_usd) AS volume, COUNT(*) AS trades
FROM dex.trades
WHERE project = '{{protocol}}'
  AND block_time >= now() - interval '{{timeframe}}'
GROUP BY 1 ORDER BY 2 DESC
LIMIT {{top_n}}
""",
        label="top pairs by volume",
        candidate_tools="[]",
        g1="Protocols",
        g2="Trading Activity",
        g3="Trading Pairs Leaderboard",
        g4="Ranked List",
        g5="Uniswap V3 top trading pairs last 4 hours",
        description="Ranks trading pairs by volume for a DEX protocol over a time window.",
        inputs_json=json.dumps([
            {"key": "protocol", "type": "protocol", "label": "Protocol", "default": "uniswap_v3", "required": False},
            {"key": "timeframe", "type": "text", "label": "Timeframe", "default": "4 hours", "required": False},
            {"key": "top_n", "type": "number", "label": "Top N", "default": 20, "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 3. Token holder count ────────────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT COUNT(DISTINCT address) AS holders
FROM (
    SELECT "to" AS address FROM erc20.evt_Transfer
    WHERE contract_address = {{token_address}}
    GROUP BY 1 HAVING SUM(value) > 0
) h
""",
        label="token holder count",
        candidate_tools="[]",
        g1="Tokens",
        g2="Holder Analytics",
        g3="Holder Count Snapshot",
        g4="Point Value",
        g5="ERC20 token holders with positive balance",
        description="Counts distinct addresses holding a positive balance of a token.",
        inputs_json=json.dumps([
            {"key": "token_address", "type": "address", "label": "Token Address", "default": None, "required": True},
            {"key": "chain", "type": "chain", "label": "Chain", "default": "ethereum", "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 4. Contract label resolution ─────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT address, name, namespace
FROM contracts
WHERE address IN ({{addresses}})
""",
        label="resolve contract labels",
        candidate_tools="[]",
        g1="Forensics",
        g2="Contract Labels",
        g3="Contract Label Lookup",
        g4="Key Value Map",
        g5="Contract addresses to human-readable labels",
        description="Resolves contract addresses to their registered names and namespaces.",
        inputs_json=json.dumps([
            {"key": "chain", "type": "chain", "label": "Chain", "default": "ethereum", "required": False},
            {"key": "addresses", "type": "text", "label": "Addresses", "default": None, "required": True},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 5. Protocol TVL by market ────────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT reserve, SUM(supply * price) AS tvl
FROM lending_protocol.balances b
JOIN prices.usd p ON b.reserve = p.contract_address
GROUP BY 1 ORDER BY 2 DESC
""",
        label="tvl by reserve",
        candidate_tools="[]",
        g1="Protocols",
        g2="Liquidity Analytics",
        g3="TVL Snapshot",
        g4="Grouped Breakdown",
        g5="Aave V3 Optimism TVL by market",
        description="Calculates total value locked per market for a lending or liquidity protocol.",
        inputs_json=json.dumps([
            {"key": "protocol", "type": "protocol", "label": "Protocol", "default": "aave_v3", "required": False},
            {"key": "chain", "type": "chain", "label": "Chain", "default": "optimism", "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 6. Revenue with moving average ───────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT date_trunc('day', block_time) AS day,
       SUM(fee) AS revenue,
       AVG(SUM(fee)) OVER (ORDER BY date_trunc('day', block_time) ROWS 6 PRECEDING) AS ma_7d
FROM protocol.fees
WHERE block_time >= now() - interval '{{timeframe}}'
GROUP BY 1 ORDER BY 1
""",
        label="revenue with moving average",
        candidate_tools="[]",
        g1="Protocols",
        g2="Revenue Analytics",
        g3="Revenue Time Series",
        g4="Multi Metric Trend",
        g5="Index Coop daily revenue and 7-day MA",
        description="Computes periodic protocol fee revenue with a moving average over time.",
        inputs_json=json.dumps([
            {"key": "protocol", "type": "protocol", "label": "Protocol", "default": "index_coop", "required": False},
            {"key": "chain", "type": "chain", "label": "Chain", "default": "ethereum", "required": False},
            {"key": "timeframe", "type": "text", "label": "Timeframe", "default": "1 year", "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 7. Wash trade detection ──────────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT tx_hash, buyer, seller, price, block_time
FROM nft.trades
WHERE nft_contract_address = {{contract_address}}
  AND buyer IN (SELECT seller FROM nft.trades WHERE nft_contract_address = {{contract_address}})
""",
        label="suspected wash trades",
        candidate_tools="[]",
        g1="Forensics",
        g2="Transaction Inspection",
        g3="Wash Trade Feed",
        g4="Event Log",
        g5="NFT wash trades for specific contract",
        description="Identifies suspected wash trades by detecting circular buyer-seller patterns.",
        inputs_json=json.dumps([
            {"key": "contract_address", "type": "address", "label": "Contract Address", "default": None, "required": True},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 8. Active users over time ────────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT date_trunc('month', block_time) AS period,
       blockchain,
       COUNT(DISTINCT "from") AS users
FROM transactions
WHERE blockchain IN ({{chains}})
GROUP BY 1, 2 ORDER BY 1
""",
        label="active users by chain",
        candidate_tools="[]",
        g1="Chains",
        g2="Activity Metrics",
        g3="Active Users Series",
        g4="Count over Time",
        g5="Monthly active users across multiple chains",
        description="Counts distinct active addresses per period across one or more blockchains.",
        inputs_json=json.dumps([
            {"key": "chains", "type": "text", "label": "Chains", "default": None, "required": True},
            {"key": "date_range", "type": "date_range", "label": "Date Range", "default": None, "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 9. Pool inflows trended ──────────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT date_trunc('day', evt_block_time) AS day,
       SUM(amount0 + amount1) AS net_inflow
FROM pool_events
WHERE event_type = 'Mint'
  AND evt_block_time >= now() - interval '{{timeframe}}'
GROUP BY 1 ORDER BY 1
""",
        label="pool inflows over time",
        candidate_tools="[]",
        g1="Protocols",
        g2="Liquidity Analytics",
        g3="Pool Inflow Series",
        g4="Count over Time",
        g5="Camelot Arbitrum pool inflows daily",
        description="Tracks net inflows into liquidity pools for a protocol over time.",
        inputs_json=json.dumps([
            {"key": "protocol", "type": "protocol", "label": "Protocol", "default": "camelot", "required": False},
            {"key": "chain", "type": "chain", "label": "Chain", "default": "arbitrum", "required": False},
            {"key": "timeframe", "type": "text", "label": "Timeframe", "default": "30 days", "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 10. Current token price ──────────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT symbol, price, decimals
FROM prices.usd_latest
WHERE contract_address = {{token_address}}
  AND blockchain = '{{chain}}'
LIMIT 1
""",
        label="latest token price",
        candidate_tools="[]",
        g1="Tokens",
        g2="Price Analytics",
        g3="Token Price Snapshot",
        g4="Point Value",
        g5="Token latest price from prices table",
        description="Returns the most recent price for a token by contract address.",
        inputs_json=json.dumps([
            {"key": "token_address", "type": "address", "label": "Token Address", "default": None, "required": True},
            {"key": "chain", "type": "chain", "label": "Chain", "default": "ethereum", "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 11. Wallet transactions ──────────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT hash, block_time, "to", value, gas_used
FROM transactions
WHERE "from" = {{address}}
ORDER BY block_time DESC
LIMIT {{limit}}
""",
        label="recent transactions from address",
        candidate_tools="[]",
        g1="Wallets",
        g2="Transaction History",
        g3="Address Transaction Feed",
        g4="Event Log",
        g5="Outgoing transactions from wallet address",
        description="Returns recent transactions sent from a wallet address.",
        inputs_json=json.dumps([
            {"key": "address", "type": "address", "label": "Address", "default": None, "required": True},
            {"key": "chain", "type": "chain", "label": "Chain", "default": "ethereum", "required": False},
            {"key": "limit", "type": "number", "label": "Limit", "default": 100, "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 12. Bridge volume by route ───────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT source_chain, dest_chain,
       SUM(amount) AS volume, COUNT(*) AS transfers
FROM bridge.transfers
WHERE token_symbol = '{{token}}'
  AND block_time >= now() - interval '{{timeframe}}'
GROUP BY 1, 2 ORDER BY 3 DESC
""",
        label="bridge volume by route",
        candidate_tools="[]",
        g1="Chains",
        g2="Economic Metrics",
        g3="Bridge Volume Leaderboard",
        g4="Ranked List",
        g5="USDC bridge volumes by source-dest pair",
        description="Ranks bridge transfer routes by volume for a token over a time window.",
        inputs_json=json.dumps([
            {"key": "token", "type": "token", "label": "Token", "default": "USDC", "required": False},
            {"key": "timeframe", "type": "text", "label": "Timeframe", "default": "30 days", "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 13. Trader segmentation ──────────────────────────────────────────────
    dspy_data.Example(
        sql="""
WITH daily AS (
    SELECT taker, date_trunc('day', block_time) AS day, SUM(amount_usd) AS vol
    FROM dex.trades
    WHERE project = '{{protocol}}'
      AND block_time >= now() - interval '{{timeframe}}'
    GROUP BY 1, 2
)
SELECT CASE
         WHEN vol >= {{whale_threshold}} THEN 'whale'
         WHEN vol >= {{pro_threshold}} THEN 'pro'
         ELSE 'retail'
       END AS tier,
       COUNT(DISTINCT taker) AS traders, SUM(vol) AS volume
FROM daily GROUP BY 1
""",
        label="trader tiers by volume",
        candidate_tools="[]",
        g1="Protocols",
        g2="Trading Activity",
        g3="Trader Volume Snapshot",
        g4="Grouped Breakdown",
        g5="DEX trade volume by whale pro retail",
        description="Segments traders into tiers by volume thresholds and aggregates activity.",
        inputs_json=json.dumps([
            {"key": "protocol", "type": "protocol", "label": "Protocol", "default": "uniswap", "required": False},
            {"key": "timeframe", "type": "text", "label": "Timeframe", "default": "90 days", "required": False},
            {"key": "whale_threshold", "type": "number", "label": "Whale Threshold", "default": 100000, "required": False},
            {"key": "pro_threshold", "type": "number", "label": "Pro Threshold", "default": 1000, "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 14. Floor price trended ──────────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT date_trunc('day', block_time) AS day,
       MIN(price) AS floor,
       AVG(MIN(price)) OVER (ORDER BY date_trunc('day', block_time) ROWS 9 PRECEDING) AS ma
FROM nft.trades
WHERE nft_contract_address = {{contract_address}}
GROUP BY 1 ORDER BY 1
""",
        label="floor price over time",
        candidate_tools="[]",
        g1="Tokens",
        g2="Price Analytics",
        g3="Floor Price Series",
        g4="Multi Metric Trend",
        g5="NFT collection floor price with moving average",
        description="Tracks floor sale price and moving average for an NFT collection over time.",
        inputs_json=json.dumps([
            {"key": "contract_address", "type": "address", "label": "Contract Address", "default": None, "required": True},
            {"key": "date_range", "type": "date_range", "label": "Date Range", "default": None, "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 15. Program activity leaderboard ─────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT program_id, SUM(compute_units) AS cu, COUNT(*) AS calls
FROM instruction_calls
WHERE block_time >= now() - interval '{{timeframe}}'
GROUP BY 1 ORDER BY 3 DESC
LIMIT {{top_n}}
""",
        label="top programs by activity",
        candidate_tools="[]",
        g1="Chains",
        g2="Activity Metrics",
        g3="Program Metrics Leaderboard",
        g4="Ranked List",
        g5="Solana top programs by CU and invocations",
        description="Ranks on-chain programs by compute consumption and call count.",
        inputs_json=json.dumps([
            {"key": "chain", "type": "chain", "label": "Chain", "default": "solana", "required": False},
            {"key": "timeframe", "type": "text", "label": "Timeframe", "default": "7 days", "required": False},
            {"key": "top_n", "type": "number", "label": "Top N", "default": 100, "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 16. NUPL ratio trended ───────────────────────────────────────────────
    dspy_data.Example(
        sql="""
WITH prices AS (
    SELECT date_trunc('hour', minute) AS t, AVG(price) AS price
    FROM prices.usd WHERE symbol = '{{token}}'
      AND minute >= now() - interval '{{timeframe}}'
    GROUP BY 1
)
SELECT t, (price - LAG(price) OVER (ORDER BY t)) / LAG(price) OVER (ORDER BY t) AS nupl
FROM prices ORDER BY 1
""",
        label="unrealized profit loss",
        candidate_tools="[]",
        g1="Tokens",
        g2="Tokenomics",
        g3="NUPL Time Series",
        g4="Ratio Trend",
        g5="Token NUPL from price data",
        description="Computes net unrealized profit/loss ratio for a token over a time window.",
        inputs_json=json.dumps([
            {"key": "token", "type": "token", "label": "Token", "default": "BTC", "required": False},
            {"key": "timeframe", "type": "text", "label": "Timeframe", "default": "60 days", "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 17. Vault balances grouped ───────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT vault_address, blockchain, SUM(balance) AS total
FROM protocol.vault_balances
GROUP BY 1, 2
""",
        label="vault balances across chains",
        candidate_tools="[]",
        g1="Protocols",
        g2="Vault Analytics",
        g3="Vault Balance Snapshot",
        g4="Grouped Breakdown",
        g5="Protocol cross-chain vault balances",
        description="Aggregates current vault balances across chains for a protocol.",
        inputs_json=json.dumps([
            {"key": "protocol", "type": "protocol", "label": "Protocol", "default": "thales", "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 18. Unique users trended ─────────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT date_trunc('day', block_time) AS day,
       COUNT(DISTINCT "from") AS users
FROM transactions
WHERE "to" = {{contract_address}}
  AND block_time >= now() - interval '{{timeframe}}'
GROUP BY 1 ORDER BY 1
""",
        label="unique users over time",
        candidate_tools="[]",
        g1="Protocols",
        g2="Usage Metrics",
        g3="Unique Users Series",
        g4="Count over Time",
        g5="Protocol daily unique interacting addresses",
        description="Tracks distinct addresses interacting with a protocol contract over time.",
        inputs_json=json.dumps([
            {"key": "protocol", "type": "protocol", "label": "Protocol", "default": "multisender", "required": False},
            {"key": "chain", "type": "chain", "label": "Chain", "default": "ethereum", "required": False},
            {"key": "timeframe", "type": "text", "label": "Timeframe", "default": "90 days", "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 19. Governance proposals ─────────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT proposal_id, title, proposer, for_votes, against_votes, status
FROM governance.proposals
WHERE dao_name = '{{dao}}'
ORDER BY start_block DESC
LIMIT {{limit}}
""",
        label="recent governance proposals",
        candidate_tools="[]",
        g1="DAOs",
        g2="Governance Activity",
        g3="Proposal Feed",
        g4="Event Log",
        g5="DAO governance proposals with tallies",
        description="Returns recent governance proposals with vote counts and status for a DAO.",
        inputs_json=json.dumps([
            {"key": "dao", "type": "protocol", "label": "DAO", "default": None, "required": True},
            {"key": "limit", "type": "number", "label": "Limit", "default": 50, "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),

    # ── 20. Wallet token flow ────────────────────────────────────────────────
    dspy_data.Example(
        sql="""
SELECT date_trunc('day', evt_block_time) AS day,
       SUM(CASE WHEN "to" = {{wallet}} THEN value ELSE 0 END) AS inflow,
       SUM(CASE WHEN "from" = {{wallet}} THEN value ELSE 0 END) AS outflow
FROM erc20.evt_Transfer
WHERE ("from" = {{wallet}} OR "to" = {{wallet}})
  AND contract_address = {{token_address}}
GROUP BY 1 ORDER BY 1
""",
        label="token flow for wallet",
        candidate_tools="[]",
        g1="Wallets",
        g2="Transaction History",
        g3="Address Transfer Volume Series",
        g4="Multi Metric Trend",
        g5="Wallet token inflow outflow over time",
        description="Tracks inflow and outflow of a token for a wallet address over time.",
        inputs_json=json.dumps([
            {"key": "wallet", "type": "address", "label": "Wallet", "default": None, "required": True},
            {"key": "token_address", "type": "address", "label": "Token Address", "default": None, "required": True},
            {"key": "chain", "type": "chain", "label": "Chain", "default": "ethereum", "required": False},
        ]),
        match=False,
        tool_id="null",
    ).with_inputs("sql", "label", "candidate_tools"),
]