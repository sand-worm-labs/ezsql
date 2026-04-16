import os
import re
import csv
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

MAX_TABLES_PER_QUERY = 6  # queries with more = dashboards, not join patterns


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
    quad_counts: Counter = Counter()
    quint_counts: Counter = Counter()
    table_counts: Counter = Counter()

    for (sql,) in rows:
        tables = extract_tables(sql)
        if len(tables) < 2:
            continue

        for table in tables:
            table_counts[table] += 1

        # Skip combo generation for large-table queries (dashboard noise)
        if len(tables) > MAX_TABLES_PER_QUERY:
            continue

        sorted_tables = sorted(tables)

        for pair in combinations(sorted_tables, 2):
            pair_counts[pair] += 1

        if len(tables) >= 3:
            for triple in combinations(sorted_tables, 3):
                triple_counts[triple] += 1

        if len(tables) >= 4:
            for quad in combinations(sorted_tables, 4):
                quad_counts[quad] += 1

        if len(tables) >= 5:
            for quint in combinations(sorted_tables, 5):
                quint_counts[quint] += 1

    with open("popular_tables.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "namespace", "table", "full_name", "count"])
        for i, (full_name, count) in enumerate(table_counts.most_common(), 1):
            parts = full_name.split(".")
            namespace = parts[-2] if len(parts) >= 2 else ""
            table = parts[-1]
            writer.writerow([i, namespace, table, full_name, count])
    print("Wrote popular_tables.csv")

    with open("popular_pairs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "table_a", "table_b", "count", "table_a_count", "table_b_count"])
        for i, ((a, b), count) in enumerate(pair_counts.most_common(), 1):
            writer.writerow([i, a, b, count, table_counts[a], table_counts[b]])
    print("Wrote popular_pairs.csv")

    with open("popular_triples.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "table_a", "table_b", "table_c", "count", "table_a_count", "table_b_count", "table_c_count"])
        for i, (triple, count) in enumerate(triple_counts.most_common(), 1):
            writer.writerow([i, *triple, count, *(table_counts[t] for t in triple)])
    print("Wrote popular_triples.csv")

    with open("popular_quads.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "table_a", "table_b", "table_c", "table_d", "count", "table_a_count", "table_b_count", "table_c_count", "table_d_count"])
        for i, (quad, count) in enumerate(quad_counts.most_common(), 1):
            writer.writerow([i, *quad, count, *(table_counts[t] for t in quad)])
    print("Wrote popular_quads.csv")

    with open("popular_quints.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "table_a", "table_b", "table_c", "table_d", "table_e", "count", "table_a_count", "table_b_count", "table_c_count", "table_d_count", "table_e_count"])
        for i, (quint, count) in enumerate(quint_counts.most_common(), 1):
            writer.writerow([i, *quint, count, *(table_counts[t] for t in quint)])
    print("Wrote popular_quints.csv")


if __name__ == "__main__":
    run()