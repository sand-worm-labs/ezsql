import os
import sys
import pandas as pd
from io import StringIO
from pathlib import Path
from dotenv import load_dotenv
from sqlmodel import text
from db.connection import get_session

load_dotenv()

def load_parquets_to_postgres():
    parquet_path = os.getenv("PARQUET_PATH", "./dataset")
    table_name   = os.getenv("TABLE_NAME", "QUERIES")

    path = Path(parquet_path)
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]

    if not files:
        print("No parquet files found.")
        sys.exit(1)
    total = 0
    with get_session() as session:
        engine = session.get_bind()
        raw_conn = engine.raw_connection()
        try:
            for file in files:
                print(f"Loading {file.name}...")
                
                df = pd.read_parquet(file)
                df = df.astype(str).replace(["None", "nan", "<NA>"], "")
                df = df.apply(lambda col: col.str.replace("\t", " ", regex=False))

                buf = StringIO()
                df.to_csv(buf, sep="\t", index=False, header=True)
                buf.seek(0)

                with raw_conn.cursor() as cur:
                    copy_sql = f'COPY "{table_name}" FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER E\'\\t\')'
                    cur.copy_expert(copy_sql, buf)
                raw_conn.commit()
                total += len(df)
                print(f"  → {len(df):,} rows written")
        finally:
            raw_conn.close()

    print(f"\nDone. {total:,} total rows written.")

if __name__ == "__main__":
    load_parquets_to_postgres()