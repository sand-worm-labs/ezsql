# scripts/load.py
import os
import sys
import pandas as pd
from io import StringIO
from pathlib import Path
from dotenv import load_dotenv
from db.connection import get_connection

load_dotenv()

def load_parquets_to_postgres():
    parquet_path  = os.getenv("PARQUET_PATH", "./dataset")
    table_name    = os.getenv("TABLE_NAME", "QUERIES")

    path  = Path(parquet_path)
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]

    if not files:
        print("No parquet files found.")
        sys.exit(1)

    conn = get_connection()
    cur  = conn.cursor()
    total = 0

    for file in files:
        print(f"Loading {file.name}...")
        df = pd.read_parquet(file)
        df = df.astype(str).replace("None", "").replace("nan", "")
        df = df.apply(lambda col: col.str.replace("\t", " ", regex=False))

        buf = StringIO()
        df.to_csv(buf, sep="\t", index=False, header=True)
        buf.seek(0)

        cur.copy_expert(
            f"COPY \"{table_name}\" FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER E'\\t')",
            buf
        )
        conn.commit()
        total += len(df)
        print(f"  → {len(df):,} rows written")

    cur.close()
    conn.close()
    print(f"\nDone. {total:,} total rows written.")

if __name__ == "__main__":
    load_parquets_to_postgres()