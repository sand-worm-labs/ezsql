import os
from pathlib import Path
import pandas as pd
from sqlmodel import text
from db.connection import get_session

def _table_name():
    return os.getenv("TABLE_NAME", "QUERIES")

def _columns_from_parquet():
    parquet_path = os.getenv("PARQUET_PATH", "../../dataset")
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

    with get_session() as session:
        session.exec(text(f'DROP TABLE IF EXISTS "{table}";'))
        session.exec(text(f'CREATE TABLE "{table}" ({cols_ddl});'))
        try:
            session.exec(text('ALTER DATABASE queries REFRESH COLLATION VERSION;'))
        except Exception as e:
            print(f"Warning: Collation refresh skipped (might not be supported or permitted): {e}")
        session.commit()
    print(f"004: {table} table created with {len(columns)} columns")

def down():
    table = _table_name()
    with get_session() as session:
        session.exec(text(f'DROP TABLE IF EXISTS "{table}";'))
        session.commit()
    print(f"004: {table} table dropped")