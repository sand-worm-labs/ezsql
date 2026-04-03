# scripts/clean_db.py
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def count(cur) -> int:
    cur.execute('SELECT COUNT(*) FROM "QUERIES"')
    return cur.fetchone()[0]

def clean():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur  = conn.cursor()

    total_before = count(cur)
    print(f"Rows before: {total_before:,}")
    print("─" * 40)

    steps = [
        # ─── DEDUPE ───
        (
            "Dedupe by query_id",
            """
            DELETE FROM "QUERIES" a
            USING "QUERIES" b
            WHERE a.query_id = b.query_id
              AND a.ctid > b.ctid
            """
        ),
        (
            "Dedupe by query_sql",
            """
            -- index on hash, not the full text
            CREATE INDEX idx_queries_sql_hash ON "QUERIES" (MD5(query_sql));
            
            -- then swap
            CREATE TABLE "QUERIES_CLEAN" AS
            SELECT DISTINCT ON (MD5(query_sql)) *
            FROM "QUERIES"
            ORDER BY MD5(query_sql), ctid;

            DROP TABLE "QUERIES";
            ALTER TABLE "QUERIES_CLEAN" RENAME TO "QUERIES";

            DROP INDEX IF EXISTS idx_queries_sql_hash;
            """
        ),
        # ─── CLEAN ───
        (
            "Drop null/empty/short SQL",
            """
            DELETE FROM "QUERIES"
            WHERE query_sql IS NULL
               OR TRIM(query_sql) = ''
               OR LENGTH(TRIM(query_sql)) < 10
            """
        ),
        (
            "Normalize owner, name, description",
            """
            UPDATE "QUERIES" SET
              owner       = LOWER(TRIM(owner)),
              name        = LOWER(TRIM(REGEXP_REPLACE(name, '(Copy of\\s*(\\(#\\d+\\))?\\s*)+', '', 'i'))),
              description = CASE
                WHEN LOWER(TRIM(description)) IN ('', 'null', 'none', 'test', 'n/a', '-') THEN NULL
                ELSE TRIM(description)
              END
            """
        ),
        (
            "Null out garbage names",
            """
            UPDATE "QUERIES"
            SET name = NULL
            WHERE LOWER(TRIM(name)) IN ('new query', 'untitled', 'query', 'test', 'unnamed query')
            """
        ),
    ]

    for label, sql in steps:
        before = count(cur)
        print(f"Running: {label}...")
        cur.execute(sql)
        conn.commit()
        after = count(cur)
        affected = cur.rowcount
        dropped  = before - after
        print(f"  → {affected:,} rows affected | {dropped:,} rows dropped | {after:,} remaining")

    print("─" * 40)
    total_after = count(cur)
    print(f"Rows after:  {total_after:,}")
    print(f"Total dropped: {total_before - total_after:,}")
    print("Done.")

    cur.close()
    conn.close()

if __name__ == "__main__":
    clean()