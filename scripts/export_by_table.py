"""
Export DB queries into working/ in two folder types:

  working/<namespace>.<table>/file.sql, file2.sql, file3.sql ...
      Queries whose SQL references ONLY this one table (single-table queries).

  working/multi/file.sql, file2.sql, file3.sql ...
      Queries whose SQL references MORE THAN ONE table.

Only tables with tier='top_80' in popular_tables_with_impact.csv are processed.
Files are split at --chunk-mb (default 40 MB).

Usage:
    python scripts/export_by_table.py [--chunk-mb 40]
"""

import os
import re
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

ROOT          = Path(__file__).parent.parent
POPULAR_CSV   = ROOT / "popular_tables_with_impact.csv"
WORKING_DIR   = ROOT / "working"
COLUMNS       = ["query_id", "name", "description", "tags", "version",
                 "parameters", "query_engine", "query_sql",
                 "is_private", "is_archived", "is_unsaved", "owner"]
INSERT_TARGET = "Query"

_TABLE_COUNT_EXPR = """
    (
        SELECT COUNT(DISTINCT m[1])
        FROM regexp_matches(
            query_sql,
            '(?:FROM|JOIN)\\s+("?[\\w]+"?\\."?[\\w]+"?(?:\\."?[\\w]+"?)?)',
            'gi'
        ) AS m
    )
"""


def table_regex(namespace: str, table: str) -> str:
    ns = re.escape(namespace)
    tb = re.escape(table)
    return f'(?:FROM|JOIN)\\s+"?{ns}"?\\."?{tb}"?'


def fmt_val(v) -> str:
    if v is None or (isinstance(v, str) and v.strip() in ("", "None", "nan", "NULL")):
        return "NULL"
    return "'" + str(v).replace("'", "''") + "'"


def row_to_insert(row: dict) -> str:
    cols = ", ".join(f'"{c}"' for c in COLUMNS)
    vals = ", ".join(fmt_val(row.get(c)) for c in COLUMNS)
    return f'INSERT INTO "{INSERT_TARGET}" ({cols}) VALUES ({vals});\n'


def chunk_name(idx: int) -> str:
    """file.sql, file2.sql, file3.sql, ..."""
    return "file.sql" if idx == 0 else f"file{idx + 1}.sql"


def write_chunks(rows: list[dict], out_dir: Path, chunk_bytes: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_idx = 0
    buf: list[str] = []
    buf_bytes = 0

    def flush():
        nonlocal chunk_idx, buf_bytes
        path = out_dir / chunk_name(chunk_idx)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(buf)
        size_mb = path.stat().st_size / 1_048_576
        print(f"    {path.name}  ({len(buf):,} rows, {size_mb:.1f} MB)")
        chunk_idx += 1
        buf.clear()
        buf_bytes = 0

    for row in rows:
        line = row_to_insert(row)
        line_bytes = len(line.encode("utf-8"))
        if buf_bytes + line_bytes > chunk_bytes and buf:
            flush()
        buf.append(line)
        buf_bytes += line_bytes

    if buf:
        flush()

    return chunk_idx


def fetch_single(conn, namespace: str, table: str) -> list[dict]:
    sel = ", ".join(COLUMNS)
    # exec_driver_sql bypasses SQLAlchemy parameter parsing so the regex
    # literal (?:FROM|JOIN) is not mistaken for a :FROM bind parameter.
    result = conn.exec_driver_sql(
        f"""
        SELECT {sel}
        FROM "queries"
        WHERE query_sql ~* %s
          AND query_sql IS NOT NULL
          AND {_TABLE_COUNT_EXPR} = 1
        """,
        (table_regex(namespace, table),),
    )
    return [dict(zip(COLUMNS, r)) for r in result]


def fetch_multi(conn, namespace: str, table: str) -> list[dict]:
    """Queries referencing this table AND at least one other table."""
    sel = ", ".join(COLUMNS)
    result = conn.exec_driver_sql(
        f"""
        SELECT {sel}
        FROM "queries"
        WHERE query_sql ~* %s
          AND query_sql IS NOT NULL
          AND {_TABLE_COUNT_EXPR} > 1
        """,
        (table_regex(namespace, table),),
    )
    return [dict(zip(COLUMNS, r)) for r in result]


def export(chunk_mb: float):
    chunk_bytes = int(chunk_mb * 1_048_576)

    popular = pd.read_csv(POPULAR_CSV)
    popular = popular[popular["tier"] == "top_80"].reset_index(drop=True)
    print(f"top_80 tables: {len(popular)}\n")

    engine = create_engine(os.environ["DATABASE_URL"])

    with engine.connect() as conn:
        for _, row in popular.iterrows():
            full_name = row["full_name"]
            namespace = str(row["namespace"])
            table     = str(row["table"])
            single_rows = fetch_single(conn, namespace, table)
            multi_rows  = fetch_multi(conn, namespace, table)

            if not single_rows and not multi_rows:
                print(f"  {full_name}: no rows — skip")
                continue

            print(f"  {full_name}: {len(single_rows):,} single / {len(multi_rows):,} multi")
            if single_rows:
                write_chunks(single_rows, WORKING_DIR / full_name / "single", chunk_bytes)
            if multi_rows:
                write_chunks(multi_rows, WORKING_DIR / full_name / "multi", chunk_bytes)

    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunk-mb", type=float, default=40.0)
    args = p.parse_args()
    export(args.chunk_mb)
