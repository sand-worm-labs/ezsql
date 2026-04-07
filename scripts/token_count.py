"""
Count tokens across all rows in the QUERIES table.
Uses tiktoken (cl100k_base) if available, else falls back to len(text)/4 estimation.
"""
from dotenv import load_dotenv
from sqlmodel import text
from db.connection import get_session

load_dotenv()

TEXT_COLUMNS = ["query_sql", "name", "description", "tags", "parameters"]

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(t: str) -> int:
        return len(enc.encode(t, disallowed_special=()))
    TOKEN_METHOD = "tiktoken cl100k_base"
except ImportError:
    def count_tokens(t: str) -> int:
        return max(1, len(t) // 4)
    TOKEN_METHOD = "estimated (chars / 4) — install tiktoken for exact counts"


def run():
    cols_sql = ", ".join(f'"{c}"' for c in TEXT_COLUMNS)
    with get_session() as session:
        rows = session.exec(
            text(f'SELECT {cols_sql} FROM "QUERIES"')
        ).all()

    print(f"Token counting method : {TOKEN_METHOD}")
    print(f"Rows fetched          : {len(rows):,}")
    print(f"Columns counted       : {', '.join(TEXT_COLUMNS)}")
    print()

    total_tokens = 0
    col_totals: dict[str, int] = {c: 0 for c in TEXT_COLUMNS}
    bar_width = 40
    n = len(rows)

    for i, row in enumerate(rows, 1):
        for col, val in zip(TEXT_COLUMNS, row):
            if val and str(val).strip():
                t = count_tokens(str(val))
                col_totals[col] += t
                total_tokens += t

        if i % 500 == 0 or i == n:
            pct = i / n
            filled = int(bar_width * pct)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(f"\r[{bar}] {i:,}/{n:,}  {total_tokens:>14,} tokens", end="", flush=True)

    print()  # newline after progress bar

    # ── Results ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TOTAL TOKEN COUNT")
    print(f"{'='*60}")
    print(f"  {'Rows processed':<28} {n:>14,}")
    print(f"  {'Total tokens':<28} {total_tokens:>14,}")
    print(f"  {'Approx cost (GPT-4o $2.50/1M)':<28} ${total_tokens / 1_000_000 * 2.50:>13,.2f}")
    print(f"  {'Approx cost (Claude $3/1M)':<28} ${total_tokens / 1_000_000 * 3.00:>13,.2f}")
    print(f"\n{'─'*60}")
    print(f"  {'Column':<28} {'Tokens':>14}  {'Share':>6}")
    print(f"{'─'*60}")
    for col in TEXT_COLUMNS:
        if not col_totals[col]:
            continue
        share = col_totals[col] / total_tokens * 100 if total_tokens else 0
        print(f"  {col:<28} {col_totals[col]:>14,}  {share:>5.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
