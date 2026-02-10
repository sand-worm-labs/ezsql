import pandas as pd
import os
import json
from tqdm import tqdm
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from query_to_text import get_full_narrative

# ------------------------
# Global synchronization
# ------------------------
file_lock = Lock()
checkpoint_lock = Lock()

def get_query_list(parquet_path: str) -> list:
    df = pd.read_parquet(parquet_path, columns=['query_id', 'query_sql'])
    return df.to_dict('records')

def load_checkpoint(checkpoint_file: str) -> set:
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return set(json.load(f))
    return set()

def save_checkpoint(checkpoint_file: str, processed_ids: set):
    with open(checkpoint_file, "w") as f:
        json.dump(list(processed_ids), f)

# ------------------------
# Worker function
# ------------------------
def process_single_query(q):
    query_id = q["query_id"]
    sql = q["query_sql"]

    try:
        narrative = get_full_narrative(sql)
    except Exception as e:
        narrative = f"Error: {str(e)}"

    output_block = (
        f"--- Query ID: {query_id} ---\n"
        f"{narrative}\n"
        f"{'-' * 50}\n\n"
    )

    return query_id, output_block

# ------------------------
# Main pipeline
# ------------------------
def process_and_save_queries(
    parquet_path: str,
    output_txt: str,
    checkpoint_file: str,
    max_workers: int = 8,   # tune this
):
    data = get_query_list(parquet_path)
    processed_ids = load_checkpoint(checkpoint_file)

    remaining_data = [q for q in data if q["query_id"] not in processed_ids]

    print(
        f"Total: {len(data)} | "
        f"Already Processed: {len(processed_ids)} | "
        f"Remaining: {len(remaining_data)}"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_query, q)
            for q in remaining_data
        ]

        with tqdm(total=len(futures), desc="Processing Intents", unit="query") as pbar:
            for future in as_completed(futures):
                query_id, output_block = future.result()

                # ---- synchronized section ----
                with file_lock:
                    with open(output_txt, "a", encoding="utf-8") as f:
                        f.write(output_block)

                with checkpoint_lock:
                    processed_ids.add(query_id)

                    # periodic checkpoint save
                    if len(processed_ids) % 100 == 0:
                        save_checkpoint(checkpoint_file, processed_ids)

                pbar.update(1)

    # final checkpoint
    save_checkpoint(checkpoint_file, processed_ids)

# ------------------------
# Execution
# ------------------------
path = "dataset/5839995-5849994.parquet"
output_file = "query_intents_output.txt"
chk_file = "checkpoint.json"

process_and_save_queries(
    path,
    output_file,
    chk_file,
    max_workers=16,  # try 4, 8, 16 depending on CPU / API limits
)
