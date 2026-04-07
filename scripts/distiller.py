from dotenv import load_dotenv
from sqlmodel import select
import tiktoken

load_dotenv()
from db.connection import get_session
from db.model.query import Query

ENC = tiktoken.get_encoding("cl100k_base")
PROMPT = """Classify each SQL query into a 2-4 word snake_case label describing its STRUCTURE.
Ignore all specifics: token names, chains, protocols, addresses, dates.
Same structure = same label. Return ONLY a JSON array of labels. No markdown."""
TOKEN_LIMIT = 900_000


def fetch(session):
    offset = 0
    chunk = 5000
    while True:
        rows = session.exec(
            select(Query)
            .where(Query.query_sql.is_not(None))
            .offset(offset)
            .limit(chunk)
        ).all()
        if not rows:
            break
        for r in rows:
            yield str(r.query_id), str(r.name or ""), str(r.query_sql)
        offset += chunk


def pack(stream):
    budget = TOKEN_LIMIT - len(ENC.encode(PROMPT)) - 300
    batch, toks = [], 0
    for qid, name, sql in stream:
        txt = f"---\nName: {name}\n```sql\n{sql}\n```\n"
        t = len(ENC.encode(txt))
        if t > budget:
            txt = ENC.decode(ENC.encode(txt)[:budget - 100])
            t = budget - 100
        if toks + t > budget and batch:
            yield batch; batch, toks = [], 0
        batch.append((qid, txt)); toks += t
    if batch: yield batch


def distiller():
    with get_session() as session:
        stream = fetch(session)
        total_q = 0
        for i, batch in enumerate(pack(stream)):
            total_q += len(batch)
            # print(batch)
            print(f"Batch {i}: {len(batch)} queries, ~{sum(len(ENC.encode(t)) for _, t in batch):,} tokens")

        print(f"\n{i+1} batches, {total_q:,} queries")


if __name__ == "__main__":
    distiller()