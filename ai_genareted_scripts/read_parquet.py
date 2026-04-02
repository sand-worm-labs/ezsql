import pandas as pd
import os
import json
from tqdm import tqdm
from threading import Lock
from test_scripts.query_to_text import get_full_narrative 

# 1. Synchronization: Create a lock for thread-safe file writing
file_lock = Lock()

def get_query_list(parquet_path: str) -> list:
    df = pd.read_parquet(parquet_path, columns=['query_id', 'query_sql'])
    return df.to_dict('records')

def load_checkpoint(checkpoint_file: str) -> set:
    """Loads a set of already processed query IDs."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return set(json.load(f))
    return set()

def save_checkpoint(checkpoint_file: str, processed_ids: set):
    """Saves the current progress to a JSON file."""
    with open(checkpoint_file, 'w') as f:
        json.dump(list(processed_ids), f)

def process_and_save_queries(parquet_path: str, output_txt: str, checkpoint_file: str):
    data = get_query_list(parquet_path)
    processed_ids = load_checkpoint(checkpoint_file)
    
    # Filter out what we already did
    remaining_data = [q for q in data if q['query_id'] not in processed_ids]
    total = len(remaining_data)
    
    print(f"Total: {len(data)} | Already Processed: {len(processed_ids)} | Remaining: {total}")

    # 2. Progress: Wrapped in tqdm for a live progress bar
    with tqdm(total=total, desc="Processing Intents", unit="query") as pbar:
        for q in remaining_data:
            query_id = q['query_id']
            sql = q['query_sql']
            
            try:
                narrative = get_full_narrative(sql)
            except Exception as e:
                narrative = f"Error: {str(e)}"

            output_block = f"--- Query ID: {query_id} ---\n{narrative}\n{'-' * 50}\n\n"
            
            # 3. Synchronization: Use the lock when writing to the shared file
            with file_lock:
                with open(output_txt, "a", encoding="utf-8") as f:
                    f.write(output_block)
            
            # Update checkpoint state
            processed_ids.add(query_id)
            pbar.update(1)

            save_checkpoint(checkpoint_file, processed_ids)

    # Final checkpoint save
    save_checkpoint(checkpoint_file, processed_ids)

# --- Execution ---
path = "dataset/5839995-5849994.parquet"
output_file = "query_intents_output.txt"
chk_file = "checkpoint.json"

process_and_save_queries(path, output_file, chk_file)