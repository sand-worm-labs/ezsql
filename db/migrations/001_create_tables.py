from sqlmodel import text
from db.connection import get_session

def up():
    with get_session() as session:
        session.exec(text("""
            CREATE TABLE IF NOT EXISTS grimoire_domains (
                domain_id    TEXT PRIMARY KEY,
                name         TEXT NOT NULL UNIQUE,
                description  TEXT NOT NULL,
                created_at   TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS grimoire_tools (
                tool_id        TEXT PRIMARY KEY,
                g1             TEXT NOT NULL,
                g2             TEXT NOT NULL,
                g3             TEXT NOT NULL,
                g4             TEXT NOT NULL,
                g5             TEXT NOT NULL,
                description    TEXT NOT NULL,
                inputs         JSONB NOT NULL DEFAULT '[]',
                source_queries TEXT[] NOT NULL DEFAULT '{}',
                usage_count    INTEGER NOT NULL DEFAULT 0,
                last_used_at   TIMESTAMPTZ,
                created_at     TIMESTAMPTZ DEFAULT NOW(),
                updated_at     TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS grimoire_tools_g1_g2_g3
                ON grimoire_tools (g1, g2, g3);
        """))
    print("001: tables created")


def down():
    with get_session() as session:
        session.exec(text("""
            DROP TABLE IF EXISTS grimoire_tools;
            DROP TABLE IF EXISTS grimoire_domains;
        """))
    print("001: tables dropped")