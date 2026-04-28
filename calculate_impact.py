import csv

INPUT_FILE = "popular_tables.csv"
OUTPUT_FILE = "popular_tables_with_impact.csv"


def main():
    rows = []
    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["count"] = int(row["count"])
            rows.append(row)

    total = sum(r["count"] for r in rows)
    cumulative = 0

    output_rows = []
    for row in rows:
        impact = row["count"] / total * 100
        cumulative += impact
        output_rows.append({
            "rank": row["rank"],
            "namespace": row["namespace"],
            "table": row["table"],
            "full_name": row["full_name"],
            "appearances": row["count"],
            "appearance_share_pct": round(impact, 4),
            "cumulative_pct": round(cumulative, 4),
            "tier": "top_80" if cumulative <= 80 else "tail_20",
        })

    top_80 = [r for r in output_rows if r["tier"] == "top_80"]
    tail_20 = [r for r in output_rows if r["tier"] == "tail_20"]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["rank", "namespace", "table", "full_name", "appearances",
                      "appearance_share_pct", "cumulative_pct", "tier"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Total appearances: {total:,}")
    print(f"Unique tables    : {len(rows):,}")
    print(f"Output written   : {OUTPUT_FILE}")
    print()
    print(f"  top_80 tier : {len(top_80):>6,} tables  →  80% of all table appearances")
    print(f"  tail_20 tier: {len(tail_20):>6,} tables  →  20% of all table appearances")
    print()
    print(f"{'Rank':<6} {'Table':<45} {'Appearances':>12}  {'Share%':>8}  {'Cumul%':>8}  {'Tier'}")
    print("-" * 95)
    for r in output_rows[:20]:
        print(f"{r['rank']:<6} {r['full_name']:<45} {r['appearances']:>12,}  "
              f"{r['appearance_share_pct']:>7.3f}%  {r['cumulative_pct']:>7.3f}%  {r['tier']}")


if __name__ == "__main__":
    main()
