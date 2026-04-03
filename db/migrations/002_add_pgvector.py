from db.connection import get_connection


def up():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;

                ALTER TABLE grimoire_tools
                    ADD COLUMN IF NOT EXISTS embedding vector(1536);

                CREATE INDEX IF NOT EXISTS grimoire_tools_embedding_idx
                    ON grimoire_tools
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
            """)
            conn.commit()
    finally:
        conn.close()
    print("002: pgvector extension and embedding column added")


def down():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DROP INDEX IF EXISTS grimoire_tools_embedding_idx;
                ALTER TABLE grimoire_tools DROP COLUMN IF EXISTS embedding;
            """)
            conn.commit()
    finally:
        conn.close()
    print("002: embedding column dropped")
