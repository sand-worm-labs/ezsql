import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlmodel import select, Session
from db.connection import engine
from db.model.query import Query


def fetch_queries(table_name: str):
    with Session(engine) as session:
        statement = select(Query.query_id, Query.query_sql, Query.description)
        results = session.exec(statement).all()
        return results


if __name__ == "__main__":
    table_name = sys.argv[1] if len(sys.argv) > 1 else "QUERIES"
    rows = fetch_queries(table_name)
    for query_id, query_sql, description in rows:
        print(f"ID: {query_id}")
        print(f"Description: {description}")
        print(f"Query: {query_sql}")
        print("-" * 60)
