import os
import torch
import sqlglot
from sqlglot import exp
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sqlglot.errors import ErrorLevel

# Authentication
os.environ["HF_TOKEN"] = "hf_"

model_name = "mrm8488/t5-base-finetuned-wikiSQL-sql-to-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate_block(text):
    """Translates a small chunk of SQL."""
    inputs = tokenizer(f"translate Sql to English: {text}", return_tensors="pt", max_length=1512, truncation=False)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_new_tokens=2000)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_full_narrative(sql_query):
    try:
        parsed = sqlglot.parse_one(sql_query, error_level=ErrorLevel.IGNORE)
        narrative = []
        
        # 1. Handle CTEs (The 'With' blocks)
        ctes = list(parsed.find_all(exp.CTE))
        for cte in ctes:
            alias = cte.alias
            # Get the SQL inside the CTE
            cte_sql = cte.this.sql()
            explanation = translate_block(cte_sql)
            narrative.append(f"Step '{alias}': {explanation}")
        
        # 2. Handle the final Select
        main_query = parsed.copy()
        # Remove the WITH part so we only translate the final SELECT
        main_query.set("with", None)
        final_explanation = translate_block(main_query.sql())
        narrative.append(f"Finally: {final_explanation}")
        
        return "\n".join(narrative)
    except Exception as e:
        return f"Could not parse query: {e}"

