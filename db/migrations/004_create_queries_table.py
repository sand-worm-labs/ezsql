import os
from pathlib import Path

import pandas as pd

from db.connection import get_connection


def _table_name():
    return os.getenv("TABLE_NAME", "QUERIES")


def _columns_from_parquet():
    parquet_path = os.getenv("PARQUET_PATH", "./dataset")
    path = Path(parquet_path)
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_path}")
    df = pd.read_parquet(files[0], engine="pyarrow")
    return list(df.columns)


def up():
    table = _table_name()
    columns = _columns_from_parquet()
    cols_ddl = ", ".join(f'"{c}" TEXT' for c in columns)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{table}";')
            cur.execute(f'CREATE TABLE "{table}" ({cols_ddl});')
            conn.commit()
    finally:
        conn.close()
    print(f"004: {table} table created")


def down():
    table = _table_name()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{table}";')
            conn.commit()
    finally:
        conn.close()
    print(f"004: {table} table dropped")
