import os
import sys
import json
import numpy as np
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# ─── UTILS ───

def serialize(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        x = x.tolist()  # fall through to list handler
    if isinstance(x, (list, dict)):
        return json.dumps(x, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    return x

def coerce_df(df: pd.DataFrame) -> pd.DataFrame:
    # apply to ALL columns — dtype check misses edge cases
    return df.apply(lambda col: col.map(serialize))

# ─── MAIN ───

def load_parquets_to_postgres():
    database_url = os.getenv("DATABASE_URL")
    parquet_path = os.getenv("PARQUET_PATH", "./dataset")
    table_name = os.getenv("TABLE_NAME", "QUERIES")

    if not database_url:
        print("DATABASE_URL not set.")
        sys.exit(1)

    engine = create_engine(database_url)
    path = Path(parquet_path)

    files = list(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        print("No parquet files found.")
        sys.exit(1)

    for i, file in enumerate(files):
        print(f"Loading {file.name}...")
        df = coerce_df(pd.read_parquet(file))
        df.to_sql(
            table_name,
            engine,
            if_exists="append" if i > 0 else "replace",
            index=False,
            method="multi",
            chunksize=1000,
        )
        print(f"  → {len(df):,} rows written")

    print("Done.")

if __name__ == "__main__":
    load_parquets_to_postgres()