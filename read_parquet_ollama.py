"""
SQL Intent Processor with Ollama (Optimized)
=============================================
Processes parquet files using Ollama for natural language generation.
Optimized for reduced disk I/O and faster throughput.
"""

import pandas as pd
import os
import json
import time
from io import StringIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from sql_narrator_ollama import SQLNarrator, OllamaConfig


# ========================
# Configuration
# ========================
CONFIG = {
    # Ollama settings
    "ollama_model": "llama3", # Small, fast, good at instruction following
    "ollama_base_url": "http://192.168.1.76:8080",
    
    # Processing settings
    "max_workers": 4,
    "batch_write_size": 100,       # Write every 100 queries
    "checkpoint_interval": 500,    # Checkpoint every 500 queries
    
    # Paths
    "parquet_path": "dataset/5839995-5849994.parquet",
    "output_file": "query_intents_output.txt",
    "checkpoint_file": "checkpoint.json",
}


# ========================
# Checkpoint Management
# ========================
def load_checkpoint(checkpoint_file: str) -> set:
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(checkpoint_file: str, processed_ids: set):
    temp_file = checkpoint_file + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(list(processed_ids), f)
    os.replace(temp_file, checkpoint_file)


# ========================
# Worker Function
# ========================
def process_single_query(query_id: str, sql_query: str, narrator: SQLNarrator) -> tuple[str, str]:
    """Process a single query."""
    try:
        narrative = narrator.narrate(sql_query)
    except Exception as e:
        narrative = f"Error: {str(e)}"
    
    output_block = (
        f"--- Query ID: {query_id} ---\n"
        f"{narrative}\n"
        f"{'-' * 50}\n\n"
    )
    
    return query_id, output_block


# ========================
# Sequential Pipeline (Optimized)
# ========================
def process_queries_sequential(
    parquet_path: str,
    output_file: str,
    checkpoint_file: str,
    ollama_model: str,
    batch_write_size: int,
    checkpoint_interval: int,
):
    """
    Process SQL queries sequentially with optimized I/O.
    """
    config = OllamaConfig(model=ollama_model)
    narrator = SQLNarrator(config)
    
    # Load data
    print("Loading data...")
    df = pd.read_parquet(parquet_path, columns=['query_id', 'query_sql'])
    processed_ids = load_checkpoint(checkpoint_file)
    
    remaining = df[~df['query_id'].isin(processed_ids)]
    print(f"Total: {len(df)} | Processed: {len(processed_ids)} | Remaining: {len(remaining)}")
    
    if len(remaining) == 0:
        print("All done!")
        return
    
    # Check Ollama
    narrator._check_ollama()
    
    # Use StringIO buffer for faster writes
    buffer = StringIO()
    buffer_count = 0
    total_processed = 0
    
    # Keep file handle open
    with open(output_file, "a", encoding="utf-8") as f:
        with tqdm(total=len(remaining), desc="Processing", unit="query") as pbar:
            for _, row in remaining.iterrows():
                query_id = row['query_id']
                sql_query = row['query_sql']
                
                try:
                    narrative = narrator.narrate(sql_query)
                except Exception as e:
                    narrative = f"Error: {str(e)}"
                
                output_block = (
                    f"--- Query ID: {query_id} ---\n"
                    f"{narrative}\n"
                    f"{'-' * 50}\n\n"
                )
                
                buffer.write(output_block)
                buffer_count += 1
                processed_ids.add(query_id)
                total_processed += 1
                
                # Batch write
                if buffer_count >= batch_write_size:
                    f.write(buffer.getvalue())
                    f.flush()
                    buffer = StringIO()
                    buffer_count = 0
                
                # Checkpoint less frequently
                if total_processed % checkpoint_interval == 0:
                    save_checkpoint(checkpoint_file, processed_ids)
                
                pbar.update(1)
        
        # Final flush
        if buffer_count > 0:
            f.write(buffer.getvalue())
            f.flush()
    
    save_checkpoint(checkpoint_file, processed_ids)
    print(f"\nDone! Processed {len(remaining)} queries.")


# ========================
# Parallel Pipeline (Optimized)
# ========================
def process_queries_parallel(
    parquet_path: str,
    output_file: str,
    checkpoint_file: str,
    ollama_model: str,
    max_workers: int,
    batch_write_size: int,
    checkpoint_interval: int,
):
    """
    Process SQL queries with parallel Ollama requests and optimized I/O.
    """
    config = OllamaConfig(model=ollama_model)
    narrator = SQLNarrator(config)
    checkpoint_lock = Lock()
    
    # Load data
    print("Loading data...")
    df = pd.read_parquet(parquet_path, columns=['query_id', 'query_sql'])
    processed_ids = load_checkpoint(checkpoint_file)
    
    remaining = df[~df['query_id'].isin(processed_ids)].to_dict('records')
    print(f"Total: {len(df)} | Processed: {len(processed_ids)} | Remaining: {len(remaining)}")
    
    if not remaining:
        print("All done!")
        return
    
    # Check Ollama
    narrator._check_ollama()
    
    # Buffers
    buffer = StringIO()
    buffer_count = 0
    total_processed = 0
    
    with open(output_file, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_query,
                    row['query_id'],
                    row['query_sql'],
                    narrator
                ): row['query_id']
                for row in remaining
            }
            
            with tqdm(total=len(futures), desc="Processing", unit="query") as pbar:
                for future in as_completed(futures):
                    query_id, output_block = future.result()
                    
                    with checkpoint_lock:
                        buffer.write(output_block)
                        buffer_count += 1
                        processed_ids.add(query_id)
                        total_processed += 1
                        
                        # Batch write
                        if buffer_count >= batch_write_size:
                            f.write(buffer.getvalue())
                            f.flush()
                            buffer = StringIO()
                            buffer_count = 0
                        
                        # Checkpoint less frequently
                        if total_processed % checkpoint_interval == 0:
                            save_checkpoint(checkpoint_file, processed_ids)
                    
                    pbar.update(1)
        
        # Final flush
        if buffer_count > 0:
            f.write(buffer.getvalue())
            f.flush()
    
    save_checkpoint(checkpoint_file, processed_ids)
    print(f"\nDone! Processed {len(remaining)} queries.")


# ========================
# Entry Point
# ========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process SQL queries to natural language")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--model", default=CONFIG["ollama_model"], help="Ollama model name")
    parser.add_argument("--workers", type=int, default=CONFIG["max_workers"], help="Max parallel workers")
    parser.add_argument("--batch", type=int, default=CONFIG["batch_write_size"], help="Batch write size")
    parser.add_argument("--checkpoint-interval", type=int, default=CONFIG["checkpoint_interval"], help="Checkpoint interval")
    parser.add_argument("--input", default=CONFIG["parquet_path"], help="Input parquet file")
    parser.add_argument("--output", default=CONFIG["output_file"], help="Output text file")
    
    args = parser.parse_args()
    
    if args.parallel:
        process_queries_parallel(
            parquet_path=args.input,
            output_file=args.output,
            checkpoint_file=CONFIG["checkpoint_file"],
            ollama_model=args.model,
            max_workers=args.workers,
            batch_write_size=args.batch,
            checkpoint_interval=args.checkpoint_interval,
        )
    else:
        process_queries_sequential(
            parquet_path=args.input,
            output_file=args.output,
            checkpoint_file=CONFIG["checkpoint_file"],
            ollama_model=args.model,
            batch_write_size=args.batch,
            checkpoint_interval=args.checkpoint_interval,
        )