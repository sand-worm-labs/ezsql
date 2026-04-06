import os
import json
import re
from psycopg2.extras import Json
from math import log
from db.connection import get_connection

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

def get_domains() -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name, description FROM grimoire_domains ORDER BY domain_id")
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

def get_registry_json() -> str:
    """Full registry as JSON string — injected into Stage 2 prompt."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tool_id, g3, g4, description, inputs,
                       array_length(source_queries, 1) AS source_count
                FROM grimoire_tools
                ORDER BY g1, g2, g3
            """)
            tools = [dict(r) for r in cur.fetchall()]
            return json.dumps(tools, indent=2, default=str)
    finally:
        conn.close()

def find_by_g3(g3: str) -> dict | None:
    """Check if a g3 tool identity already exists in the registry."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM grimoire_tools WHERE g3 = %s LIMIT 1
            """, (g3,))
            result = cur.fetchone()
            return dict(result) if result else None
    finally:
        conn.close()

def upsert_tool(tool_data: dict, source_query_id: str) -> str:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tool_id, source_queries FROM grimoire_tools WHERE g3 = %s LIMIT 1",
                (tool_data["g3"],)
            )
            existing = cur.fetchone()

            if existing:
                tool_id = existing["tool_id"]
                cur.execute("""
                    UPDATE grimoire_tools
                    SET source_queries = array_append(source_queries, %s),
                        updated_at = NOW()
                    WHERE tool_id = %s
                    AND NOT (%s = ANY(source_queries))
                """, (source_query_id, tool_id, source_query_id))
                conn.commit()
                return tool_id

            tool_id = slugify(tool_data["g3"])
            description = tool_data.get("g5", "")
            cur.execute("""
                INSERT INTO grimoire_tools
                    (tool_id, g1, g2, g3, g4, g5, description, inputs, source_queries)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, ARRAY[%s])
            """, (
                tool_id,
                tool_data["g1"], tool_data["g2"], tool_data["g3"],
                tool_data["g4"], tool_data["g5"],
                description,
                Json(tool_data["inputs"]),
                source_query_id,
            ))
            conn.commit()
            return tool_id
    finally:
        conn.close()

def increment_usage(tool_id: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE grimoire_tools
                SET usage_count = usage_count + 1,
                    last_used_at = NOW()
                WHERE tool_id = %s
            """, (tool_id,))
            conn.commit()
    finally:
        conn.close()

def popularity_score(usage_count: int, source_count: int) -> float:
    """Combined popularity score for re-ranking after cosine retrieval."""
    return log(usage_count + 1) * 0.6 + log(source_count + 1) * 0.4