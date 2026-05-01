import os
from dotenv import load_dotenv
from sqlmodel import text
from db.connection import get_session

load_dotenv()

def count(session) -> int:
    """Helper to get current row count using the active session."""
    result = session.exec(text('SELECT COUNT(*) FROM "queries"'))
    return result.one()[0]

def clean():
    with get_session() as session:
        total_before = count(session)
        print(f"Rows before: {total_before:,}\n" + "─" * 40)
        steps = [
            (
                "Dedupe by query_id",
                """
                DELETE FROM "queries" a
                USING "queries" b
                WHERE a.query_id = b.query_id
                  AND a.ctid > b.ctid
                """
            ),
            (
                "Dedupe by query_sql",
                """
                CREATE INDEX IF NOT EXISTS idx_queries_sql_hash ON "queries" (MD5(query_sql));
                
                CREATE TABLE "QUERIES_CLEAN" AS
                SELECT DISTINCT ON (MD5(query_sql)) *
                FROM "queries"
                ORDER BY MD5(query_sql), ctid;

                DROP TABLE "queries";
                ALTER TABLE "QUERIES_CLEAN" RENAME TO "queries";
                """
            ),
            (
                "Drop null/empty/short SQL",
                """
                DELETE FROM "queries"
                WHERE query_sql IS NULL
                   OR TRIM(query_sql) = ''
                   OR LENGTH(TRIM(query_sql)) < 10
                """
            ),
            (
                "Normalize owner, name, description",
                """
                UPDATE "queries" SET
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
                UPDATE "queries"
                SET name = NULL
                WHERE LOWER(TRIM(name)) IN ('new query', 'untitled', 'query', 'test', 'unnamed query')
                """
            ),
            (
                "Final purge of null SQL",
                """
                DELETE FROM "queries"
                WHERE query_sql IS NULL 
                   OR TRIM(query_sql) = ''
                """
            ),
        ]

        for label, sql in steps:
            before = count(session)
            print(f"Running: {label}...")
            
            session.exec(text(sql))
            session.commit()  
 
            after = count(session)
            dropped = before - after
            print(f"  → {dropped:,} rows dropped | {after:,} remaining")

        total_after = count(session)
        print(f"{'─' * 40}\nRows after:  {total_after:,}\nTotal dropped: {total_before - total_after:,}\nDone.")

if __name__ == "__main__":
    clean()