import os
import sys
import psycopg2
import pandas as pd
from io import StringIO
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def load_parquets_to_postgres():
    database_url = os.getenv("DATABASE_URL")
    parquet_path = os.getenv("PARQUET_PATH", "./dataset")
    table_name = os.getenv("TABLE_NAME", "QUERIES")

    path = Path(parquet_path)
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]

    if not files:
        print("No parquet files found.")
        sys.exit(1)

    conn = psycopg2.connect(database_url)
    cur = conn.cursor()

    seen_ids  = set()
    seen_sqls = set()
    first = True
    total_written = 0
    total_dupes = 0

    for file in files:
        print(f"Loading {file.name}...")
        df = pd.read_parquet(file)

        before = len(df)
        df = df[~df["query_id"].isin(seen_ids)]
        df = df[~df["query_sql"].isin(seen_sqls)]
        df = df.drop_duplicates(subset=["query_id"],  keep="first")
        df = df.drop_duplicates(subset=["query_sql"], keep="first")
        dupes = before - len(df)
        total_dupes += dupes

        seen_ids.update(df["query_id"].tolist())
        seen_sqls.update(df["query_sql"].tolist())

        if df.empty:
            print(f"  → skipped (all dupes)")
            continue

        df = df.astype(str).replace("None", "").replace("nan", "")
        
        # strip tabs from all fields so they don't break the delimiter
        df = df.apply(lambda col: col.str.replace("\t", " ", regex=False))

        buf = StringIO()
        df.to_csv(buf, sep="\t", index=False, header=True)
        buf.seek(0)

        if first:
            cols = ", ".join(f'"{c}" TEXT' for c in df.columns)
            cur.execute(f'DROP TABLE IF EXISTS "{table_name}";')
            cur.execute(f'CREATE TABLE "{table_name}" ({cols});')
            first = False

        cur.copy_expert(
            f'COPY "{table_name}" FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER E\'\\t\')',
            buf
        )
        conn.commit()
        total_written += len(df)
        print(f"  → {len(df):,} written, {dupes:,} dupes dropped")

    cur.close()
    conn.close()
    print(f"\nDone. {total_written:,} rows written, {total_dupes:,} total dupes dropped.")

if __name__ == "__main__":
    load_parquets_to_postgres()