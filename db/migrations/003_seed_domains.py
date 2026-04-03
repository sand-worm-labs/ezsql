from db.connection import get_connection

DOMAINS = [
    ("protocols",   "Protocols",   "DEX, lending, AMM, staking, governance — any on-chain protocol interaction"),
    ("tokens",      "Tokens",      "Supply, price, holders, benchmarks — token-centric analysis"),
    ("wallets",     "Wallets",     "Address profiling, PnL, fund flows, user behavior"),
    ("chains",      "Chains",      "Network activity, L2s, bridges, contract monitoring"),
    ("network",     "Network",     "Chain-level metrics — transaction counts, gas, block data"),
    ("daos",        "DAOs",        "Governance proposals, treasury, voting"),
    ("forensics",   "Forensics",   "Fund tracing, cluster analysis, address attribution"),
    ("derivatives", "Derivatives", "Perps, options, funding rates, CEX vs DEX comparison"),
]


def up():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for domain_id, name, description in DOMAINS:
                cur.execute("""
                    INSERT INTO grimoire_domains (domain_id, name, description)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (domain_id) DO NOTHING
                """, (domain_id, name, description))
            conn.commit()
    finally:
        conn.close()
    print("003: domains seeded")


def down():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM grimoire_domains")
            conn.commit()
    finally:
        conn.close()
    print("003: domains cleared")
