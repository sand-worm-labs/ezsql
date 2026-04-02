import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.optimizer import optimize
from rich import print

def pre_optimize_trino(sql_string):
    """
    Normalizes Trino SQL into a canonical form. 
    This flattens subqueries and simplifies logic to help the intent parser.
    """
    try:
        # We use dialect="trino" for both read and write
        # optimize() simplifies predicates, pushes down filters, and removes redundant logic
        optimized = optimize(sql_string, dialect="trino")
        return optimized
    except Exception as e:
        # Fallback to simple parsing if full optimization fails
        return sqlglot.parse_one(sql_string, read="trino")

def parse_sql_to_intent(parquet_path):
    df = pd.read_parquet(parquet_path)
    intent_results = []

    for _, row in df.head(10).iterrows():
        sql_string = row['query_sql']
        query_id = row['query_id']

        try:
            # STEP 1: Optimize before parsing
            expression = pre_optimize_trino(sql_string)

            # STEP 2: Extract Reconstructible Elements
            tables = [t.sql() for t in expression.find_all(exp.Table)]
            
            # Look for specific Trino behaviors (e.g. UNNEST, Window Functions)
            has_unnest = any(isinstance(node, exp.Explode) or "UNNEST" in node.sql().upper() 
                             for node in expression.find_all(exp.Table))

            intent_results.append({
                "query_id": query_id,
                "intent_type": expression.key.upper(),
                "has_unnest": has_unnest,
                "tables": sorted(list(set(tables))),
                "clean_sql": expression.sql(dialect="trino", pretty=True)
            })

        except Exception:
            continue

    return intent_results

if __name__ == "__main__":
    FILE_PATH = "dataset/5839995-5849994.parquet"
    parsed_queries = parse_sql_to_intent(FILE_PATH)
    print(parsed_queries)