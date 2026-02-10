import sqlglot
from sqlglot import exp
from transformers import pipeline

# Load a model that is strong at summarization and logic
explainer = pipeline("text2text-generation", model="gaussalgo/T5-SQL-Explainer")

def translate_with_cte(sql_string):
    expression = sqlglot.parse_one(sql_string, read="trino")
    
    # 1. Extract all WITH clauses
    ctes = expression.find_all(exp.CTE)
    narrative_steps = []
    
    for cte in ctes:
        cte_name = cte.alias
        # We simplify the CTE SQL so the AI can focus on the core logic
        cte_logic = cte.this.sql(pretty=True)
        
        # Explain this specific block
        step_desc = explainer(f"Summarize this logic: {cte_logic}", max_length=64)[0]['generated_text']
        narrative_steps.append(f"In the step named '{cte_name}', the query {step_desc}.")

    # 2. Extract the main SELECT (the "Result")
    # We remove the CTEs from the string to get just the final operation
    main_query = expression.copy()
    main_query.set("with", None)
    final_desc = explainer(f"Explain the final result: {main_query.sql()}", max_length=64)[0]['generated_text']

    # 3. Assemble the Full Context
    full_narrative = " ".join(narrative_steps) + f" Finally, it {final_desc}."
    return full_narrative

# --- Example ---
complex_sql = """
WITH volume_calc AS (
    SELECT token, sum(amount) as total FROM trades GROUP BY 1
)
SELECT * FROM volume_calc WHERE total > 1000
"""

print(translate_with_cte(complex_sql))
# Output: In the step named 'volume_calc', the query calculates total sum of amount per token. 
# Finally, it selects tokens where the total volume is over 1000.