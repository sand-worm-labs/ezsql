import os
from dotenv import load_dotenv
from sqlmodel import select
import tiktoken
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
from db.connection import get_session
from db.model.query import Query

ENC = tiktoken.get_encoding("cl100k_base")
PROMPT = """
For each numbered query below, return ONE structural label on its own line.

RULES:
1. Labels are 4-8 words, snake_case.
2. NEVER include token names, chain names, protocol names, addresses, or dates.
3. Return EXACTLY one label per query, one per line.
4. No numbering, no bullets, no explanation.
5. Describe WHAT the query computes structurally, not just the topic.
6. Use consistent labels for similar queries even if they differ in protocol or chain.
7. Each input is formatted as: Q<n>. Name: <title> | SQL: <abbreviated_sql>

GOOD examples:
Q1. Name: usage | SQL: WITH allFeePayments AS (...) SELECT day, daily_trades, daily_users ...
→ daily_bot_trades_and_users_timeseries

Q2. Name: users activity | SQL: WITH base_borrowing AS (...) SELECT address, collateral, existing_debt, borrowing_apr, LTV WHERE existing_debt > 0
→ active_lending_positions_with_collateral_ratio

Q3. Name: address labels for 0x68* | SQL: SELECT * FROM labels.addresses WHERE varbinary_substring(address, 1, 1) = 0x68
→ address_labels_by_prefix_filter

Q4. Name: zbtc holders | SQL: WITH latest_balances AS (...) inflow AS (...) outflow AS (...) SELECT address, amount, percent, netflow_24h, netflow_7d ORDER BY amount DESC
→ top_holders_with_netflow_breakdown

Q5. Name: eigenlayer - list of strategies | SQL: WITH deposit AS (...) SELECT contract_address, symbol, decimals, strategy
→ staking_strategy_list_with_token_metadata

BAD examples (DO NOT produce labels like these):
✗ eigenlayer_strategy_list          — contains protocol name
✗ zbtc_holder_balances              — contains token name
✗ query_about_user_activity         — describes topic, not structure
✗ data                              — too vague, under 4 words
"""

TOKEN_LIMIT = 40_000
llm = ChatOpenRouter(
model="google/gemini-2.5-flash-lite",
temperature=0,
openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
) 

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
    import time
    with get_session() as session:
        stream = fetch(session)
        total_q = 0
        t0 = time.time()

        for i, batch in enumerate(pack(stream)):
            total_q += len(batch)
            payload = "\n".join(f"Q{j+1}. {txt}" for j, (_, txt) in enumerate(batch))

            response = llm.invoke([
                SystemMessage(content=PROMPT),
                HumanMessage(content=f"Classify these {len(batch)} queries. Return exactly {len(batch)} labels, one per line.\n\n{payload}")
            ])

            ids = [qid for qid, _ in batch]
            labels = [l.strip().lower().replace(" ", "_") for l in response.content.strip().split("\n") if l.strip()]

            if len(labels) < len(batch):
                labels += ["error"] * (len(batch) - len(labels))

            results = list(zip(ids, labels[:len(batch)]))

            with open("labels.tsv", "a") as f:
                for qid, label in results:
                    f.write(f"{qid}\t{label}\n")

            if i % 100 == 0:
                elapsed = time.time() - t0
                rate = total_q / elapsed if elapsed > 0 else 0
                remaining = (853_000 - total_q) / rate if rate > 0 else 0
                print(f"Batch {i}: {total_q:,} queries | {rate:.0f} q/s | ETA: {remaining/3600:.1f}h")

        print(f"\nDone: {total_q:,} queries in {(time.time()-t0)/60:.0f}min → labels.tsv")


if __name__ == "__main__":
    distiller()