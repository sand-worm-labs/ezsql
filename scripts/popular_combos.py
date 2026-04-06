import os
import re
from collections import Counter
from itertools import combinations
from dotenv import load_dotenv
from sqlmodel import text
from db.connection import get_session

load_dotenv()

TABLE_PATTERN = re.compile(
    r'(?:FROM|JOIN)\s+'
    r'("?[\w]+"?\."?[\w]+"?'
    r'(?:\."?[\w]+"?)?)',
    re.IGNORECASE,
)

def extract_tables(sql: str) -> set[str]:
    matches = TABLE_PATTERN.findall(sql)
    return {m.lower().replace('"', '').strip() for m in matches if len(m) > 3}


def run():
    with get_session() as session:
        rows = session.exec(
            text('SELECT query_sql FROM "QUERIES" WHERE query_sql IS NOT NULL')
        ).all()

    pair_counts: Counter = Counter()
    triple_counts: Counter = Counter()

    for (sql,) in rows:
        tables = extract_tables(sql)
        if len(tables) < 2:
            continue

        for pair in combinations(sorted(tables), 2):
            pair_counts[pair] += 1

        if len(tables) >= 3:
            for triple in combinations(sorted(tables), 3):
                triple_counts[triple] += 1

    print(f"\n{'='*90}")
    print(f"TOP 30 TABLE PAIRS")
    print(f"{'='*90}")
    print(f"{'Rank':<6} {'Table A':<40} {'Table B':<40} {'Count':>8}")
    print("-" * 96)
    for i, ((a, b), count) in enumerate(pair_counts.most_common(30), 1):
        print(f"{i:<6} {a:<40} {b:<40} {count:>8,}")

    print(f"\n{'='*90}")
    print(f"TOP 20 TABLE TRIPLES")
    print(f"{'='*90}")
    print(f"{'Rank':<6} {'Tables':<80} {'Count':>8}")
    print("-" * 90)
    for i, (triple, count) in enumerate(triple_counts.most_common(20), 1):
        print(f"{i:<6} {' + '.join(triple):<80} {count:>8,}")


if __name__ == "__main__":
    run()