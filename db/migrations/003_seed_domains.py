from sqlmodel import text
from db.connection import get_session

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
    with get_session() as session:
        for domain_id, name, description in DOMAINS:
            statement = text("""
                INSERT INTO grimoire_domains (domain_id, name, description)
                VALUES (:domain_id, :name, :description)
                ON CONFLICT (domain_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description;
            """)

            session.exec(
                statement,
                params={"domain_id": domain_id, "name": name, "description": description}
            )
        session.commit()
    print("003: domains seeded")


def down():
    with get_session() as session:
        session.exec(text("TRUNCATE TABLE grimoire_domains CASCADE;"))
        session.commit()
    print("003: domains cleared")