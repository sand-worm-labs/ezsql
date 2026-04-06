import argparse
from dotenv import load_dotenv

load_dotenv()

from db.connection import get_session
from db.model.query import Query
from dotenv import load_dotenv
from sqlmodel import select
from tool_extractor.extractor import run_grimoire_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max queries to process")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N queries")
    args = parser.parse_args()

    with get_session() as session:
        stmt = select(Query).offset(args.offset)
        if args.limit:
            stmt = stmt.limit(args.limit)
        queries = session.exec(stmt).all()
        print(f"Processing {len(queries)} queries...\n")

        for i, q in enumerate(queries, 1):
            title = q.name or q.query_id
            # print(f"[{i}/{len(queries)}] {title}")
            # try:
            #     result = run_grimoire_pipeline(
            #         user_id="extractor",
            #         title=title,
            #         sql=q['query_sql'],
            #     )
            #     tools = result.get("processed_tools", [])
            #     print(f"  → {len(tools)} tool(s) extracted")
            # except Exception as e:
            #     print(f"  ✗ Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
