import pandas as pd
import os
import json
import torch
import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ========================
# Configuration
# ========================
MODEL_NAME = "mrm8488/t5-base-finetuned-wikiSQL-sql-to-en"
BATCH_SIZE = 8  # Adjust based on available RAM
MAX_INPUT_LENGTH = 100512  # Truncate long queries
MAX_OUTPUT_LENGTH = 510
# ========================
# Model Loading (once)
# ========================
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()  # Set to inference mode

# Use half precision if available (saves ~50% memory)
if torch.cuda.is_available():
    model = model.half().cuda()
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Model loaded on {DEVICE}")


# ========================
# Translation Functions
# ========================
def translate_batch(texts: list[str]) -> list[str]:
    """Translate a batch of SQL fragments efficiently."""
    prompts = [f"translate Sql to English: {t}" for t in texts]
    
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=True
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_OUTPUT_LENGTH,
            num_beams=1,  # Greedy decoding is faster
            do_sample=False
        )
    
    # Free memory immediately
    del inputs
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def get_sql_components(sql_query: str) -> list[tuple[str, str]]:
    """
    Extract translatable components from SQL.
    Returns list of (label, sql_fragment) tuples.
    """
    try:
        parsed = sqlglot.parse_one(sql_query, error_level=ErrorLevel.IGNORE)
        components = []
        
        # Extract CTEs
        for cte in parsed.find_all(exp.CTE):
            alias = cte.alias
            cte_sql = cte.this.sql()
            components.append((f"Step '{alias}'", cte_sql))
        
        # Extract main query (without CTEs)
        main_query = parsed.copy()
        main_query.set("with", None)
        components.append(("Finally", main_query.sql()))
        
        return components
    except Exception as e:
        # Fallback: treat entire query as one component
        return [("Query", sql_query)]


def build_narrative(query_id: str, sql_query: str, translations: dict[str, str]) -> str:
    """Build narrative from pre-computed translations."""
    try:
        components = get_sql_components(sql_query)
        lines = []
        for label, sql_frag in components:
            key = f"{query_id}::{label}"
            if key in translations:
                lines.append(f"{label}: {translations[key]}")
        return "\n".join(lines) if lines else "Could not generate narrative"
    except Exception as e:
        return f"Error: {str(e)}"


# ========================
# Checkpoint Management
# ========================
def load_checkpoint(checkpoint_file: str) -> set:
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(checkpoint_file: str, processed_ids: set):
    with open(checkpoint_file, "w") as f:
        json.dump(list(processed_ids), f)


# ========================
# Main Pipeline
# ========================
def process_queries(
    parquet_path: str,
    output_txt: str,
    checkpoint_file: str,
    batch_size: int = BATCH_SIZE
):
    # Load data
    df = pd.read_parquet(parquet_path, columns=['query_id', 'query_sql'])
    processed_ids = load_checkpoint(checkpoint_file)
    
    # Filter remaining
    remaining = df[~df['query_id'].isin(processed_ids)].to_dict('records')
    
    print(f"Total: {len(df)} | Already Processed: {len(processed_ids)} | Remaining: {len(remaining)}")
    
    if not remaining:
        print("All queries already processed!")
        return
    
    # Phase 1: Extract all SQL components to translate
    print("Extracting SQL components...")
    all_fragments = []  # [(key, sql_fragment), ...]
    query_map = {}  # query_id -> sql_query
    
    for row in tqdm(remaining, desc="Parsing"):
        query_id = row['query_id']
        sql_query = row['query_sql']
        query_map[query_id] = sql_query
        
        components = get_sql_components(sql_query)
        for label, sql_frag in components:
            key = f"{query_id}::{label}"
            all_fragments.append((key, sql_frag))
    
    print(f"Total fragments to translate: {len(all_fragments)}")
    
    # Phase 2: Batch translate all fragments
    print("Translating fragments...")
    translations = {}
    
    for i in tqdm(range(0, len(all_fragments), batch_size), desc="Translating"):
        batch = all_fragments[i:i + batch_size]
        keys = [k for k, _ in batch]
        sqls = [s for _, s in batch]
        
        results = translate_batch(sqls)
        
        for key, result in zip(keys, results):
            translations[key] = result
    
    # Phase 3: Build narratives and write output
    print("Writing output...")
    with open(output_txt, "a", encoding="utf-8") as f:
        for row in tqdm(remaining, desc="Writing"):
            query_id = row['query_id']
            sql_query = row['query_sql']
            
            narrative = build_narrative(query_id, sql_query, translations)
            
            output_block = (
                f"--- Query ID: {query_id} ---\n"
                f"{narrative}\n"
                f"{'-' * 50}\n\n"
            )
            f.write(output_block)
            
            processed_ids.add(query_id)
            
            # Periodic checkpoint
            if len(processed_ids) % 100 == 0:
                save_checkpoint(checkpoint_file, processed_ids)
    
    # Final checkpoint
    save_checkpoint(checkpoint_file, processed_ids)
    print("Done!")


# ========================
# Entry Point
# ========================
if __name__ == "__main__":
    process_queries(
        parquet_path="dataset/5839995-5849994.parquet",
        output_txt="query_intents_output.txt",
        checkpoint_file="checkpoint.json",
        batch_size=8  # Lower this if still OOM
    )