import json
import re
from math import log
from typing import Optional
from sqlalchemy.dialects.postgresql import insert
from sqlmodel import text, select
from db.connection import get_session

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

def get_domains() -> list[dict]:
    with get_session() as session:
        statement = text("SELECT name, description FROM grimoire_domains ORDER BY domain_id")
        result = session.exec(statement)
        return [dict(r) for r in result.mappings()]

def get_registry_json() -> str:
    """Full registry as JSON string — injected into Stage 2 prompt."""
    query = text("""
        SELECT tool_id, g3, g4, description, inputs,
               cardinality(source_queries) AS source_count
        FROM grimoire_tools
        ORDER BY g1, g2, g3
    """)
    with get_session() as session:
        result = session.exec(query)
        tools = [dict(r) for r in result.mappings()]
        return json.dumps(tools, indent=2, default=str)

def find_by_g3(g3: str) -> dict | None:
    """Check if a g3 tool identity already exists in the registry."""
    query = text("SELECT * FROM grimoire_tools WHERE g3 = :g3 LIMIT 1")
    with get_session() as session:
        result = session.exec(query, params={"g3": g3}).mappings().first()
        return dict(result) if result else None

def upsert_tool(tool_data: dict, source_query_id: str) -> str:
    """SQLModel implementation of tool upsert with array handling."""
    with get_session() as session:
        # 1. Check for existing tool
        find_query = text("SELECT tool_id, source_queries FROM grimoire_tools WHERE g3 = :g3")
        existing = session.exec(find_query, params={"g3": tool_data["g3"]}).mappings().first()

        if existing:
            tool_id = existing["tool_id"]
            # 2. Update source_queries array if ID not already present
            update_query = text("""
                UPDATE grimoire_tools
                SET source_queries = array_append(source_queries, :qid),
                    updated_at = NOW()
                WHERE tool_id = :tid
                AND NOT (:qid = ANY(source_queries))
            """)
            session.exec(update_query, params={"qid": source_query_id, "tid": tool_id})
            session.commit()
            return tool_id

        # 3. Insert new tool
        tool_id = slugify(tool_data["g3"])
        insert_query = text("""
            INSERT INTO grimoire_tools
                (tool_id, g1, g2, g3, g4, g5, description, inputs, source_queries)
            VALUES (:tid, :g1, :g2, :g3, :g4, :g5, :desc, :inputs, ARRAY[:qid])
        """)
        session.exec(insert_query, params={
            "tid": tool_id,
            "g1": tool_data["g1"],
            "g2": tool_data["g2"],
            "g3": tool_data["g3"],
            "g4": tool_data["g4"],
            "g5": tool_data["g5"],
            "desc": tool_data.get("g5", ""),
            "inputs": json.dumps(tool_data["inputs"]), 
            "qid": source_query_id
        })
        session.commit()
        return tool_id

def increment_usage(tool_id: str):
    with get_session() as session:
        query = text("""
            UPDATE grimoire_tools
            SET usage_count = usage_count + 1,
                last_used_at = NOW()
            WHERE tool_id = :tid
        """)
        session.exec(query, params={"tid": tool_id})
        session.commit()

def popularity_score(usage_count: int, source_count: int) -> float:
    return log(usage_count + 1) * 0.6 + log(source_count + 1) * 0.4