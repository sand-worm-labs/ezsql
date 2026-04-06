from sqlmodel import text
from db.connection import get_session

def up():
    with get_session() as session:
        session.exec(text("""
            -- Enable the pgvector extension to allow vector data types
            CREATE EXTENSION IF NOT EXISTS vector;

            -- Add a vector column for 1536-dimensional embeddings (standard for OpenAI/modern LLMs)
            ALTER TABLE grimoire_tools 
                ADD COLUMN IF NOT EXISTS embedding vector(1536);

            -- Create an IVFFlat index for efficient similarity searching
            -- Note: 'lists' should typically be sqrt(rows) or rows/1000 for large datasets
            CREATE INDEX IF NOT EXISTS grimoire_tools_embedding_idx 
                ON grimoire_tools 
                USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
        """))
        session.commit()
    print("002: pgvector extension and embedding column added")


def down():
    with get_session() as session:
        session.exec(text("""
            DROP INDEX IF EXISTS grimoire_tools_embedding_idx;
            ALTER TABLE grimoire_tools DROP COLUMN IF EXISTS embedding;
        """))
        session.commit()
    print("002: embedding column dropped")