"""
SQL Intent Processor with Ollama
=================================
Processes parquet files using Ollama for natural language generation.
Includes checkpointing, rate limiting, and graceful fallback.
"""

import pandas as pd
import os
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore

from sql_narrator_ollama import SQLNarrator, OllamaConfig, narrate_sql


# ========================
# Configuration
# ========================
CONFIG = {
    # Ollama settings
    "ollama_model": "llama3",  # Change based on your VRAM
    "ollama_base_url": "http://192.168.1.76:8080",
    
    # Processing settings
    "max_workers": 8,           # Concurrent requests to Ollama
    "batch_write_size":2,     # Flush to disk every N queries
    "requests_per_second": 1,  # Rate limit (adjust based on your hardware)
    
    # Paths
    "parquet_path": "dataset/5839995-5849994.parquet",
    "output_file": "query_intents_output.txt",
    "checkpoint_file": "checkpoint.json",
}


# ========================
# Rate Limiter
# ========================
class RateLimiter:
    """Simple token bucket rate limiter."""
    
    def __init__(self, requests_per_second: float):
        self.rate = requests_per_second
        self.last_request = 0.0
        self.lock = Lock()
    
    def acquire(self):
        with self.lock:
            now = time.time()
            min_interval = 1.0 / self.rate
            elapsed = now - self.last_request
            
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            self.last_request = time.time()


# ========================
# Checkpoint Management
# ========================
def load_checkpoint(checkpoint_file: str) -> set:
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(checkpoint_file: str, processed_ids: set):
    # Atomic write
    temp_file = checkpoint_file + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(list(processed_ids), f)
    os.replace(temp_file, checkpoint_file)


# ========================
# Worker Function
# ========================
def process_single_query(query_id: str, sql_query: str, narrator: SQLNarrator, rate_limiter: RateLimiter) -> tuple[str, str]:
    """Process a single query with rate limiting."""
    rate_limiter.acquire()
    
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
# Main Pipeline
# ========================
def process_queries_parallel(
    parquet_path: str,
    output_file: str,
    checkpoint_file: str,
    ollama_model: str,
    max_workers: int,
    batch_write_size: int,
    requests_per_second: float,
):
    """
    Process SQL queries with parallel Ollama requests.
    """
    # Initialize
    config = OllamaConfig(model=ollama_model)
    narrator = SQLNarrator(config)
    rate_limiter = RateLimiter(requests_per_second)
    file_lock = Lock()
    checkpoint_lock = Lock()
    
    # Load data
    print("Loading data...")
    df = pd.read_parquet(parquet_path, columns=['query_id', 'query_sql'])
    processed_ids = load_checkpoint(checkpoint_file)
    
    # Filter remaining
    remaining = df[~df['query_id'].isin(processed_ids)].to_dict('records')
    
    print(f"Total: {len(df)} | Processed: {len(processed_ids)} | Remaining: {len(remaining)}")
    
    if not remaining:
        print("All done!")
        return
    
    # Check Ollama once
    narrator._check_ollama()
    
    # Process with thread pool
    write_buffer = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_query,
                row['query_id'],
                row['query_sql'],
                narrator,
                rate_limiter
            ): row['query_id']
            for row in remaining
        }
        
        with tqdm(total=len(futures), desc="Processing", unit="query") as pbar:
            for future in as_completed(futures):
                query_id, output_block = future.result()
                
                with checkpoint_lock:
                    processed_ids.add(query_id)
                    write_buffer.append(output_block)
                    
                    # Batch write
                    if len(write_buffer) >= batch_write_size:
                        with file_lock:
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.writelines(write_buffer)
                        write_buffer.clear()
                        save_checkpoint(checkpoint_file, processed_ids)
                
                pbar.update(1)
    
    # Final flush
    if write_buffer:
        with open(output_file, "a", encoding="utf-8") as f:
            f.writelines(write_buffer)
    
    save_checkpoint(checkpoint_file, processed_ids)
    print(f"\nDone! Processed {len(remaining)} queries.")


def process_queries_sequential(
    parquet_path: str,
    output_file: str,
    checkpoint_file: str,
    ollama_model: str,
    batch_write_size: int,
):
    """
    Process SQL queries sequentially (simpler, good for slower hardware).
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
    
    write_buffer = []
    
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
            
            write_buffer.append(output_block)
            processed_ids.add(query_id)
            
            # Batch write
            if len(write_buffer) >= batch_write_size:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.writelines(write_buffer)
                write_buffer.clear()
                save_checkpoint(checkpoint_file, processed_ids)
            
            pbar.update(1)
    
    # Final flush
    if write_buffer:
        with open(output_file, "a", encoding="utf-8") as f:
            f.writelines(write_buffer)
    
    save_checkpoint(checkpoint_file, processed_ids)
    print(f"\nDone!")


# ========================
# Entry Point
# ========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process SQL queries to natural language")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--model", default=CONFIG["ollama_model"], help="Ollama model name")
    parser.add_argument("--workers", type=int, default=CONFIG["max_workers"], help="Max parallel workers")
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
            batch_write_size=CONFIG["batch_write_size"],
            requests_per_second=CONFIG["requests_per_second"],
        )
    else:
        process_queries_sequential(
            parquet_path=args.input,
            output_file=args.output,
            checkpoint_file=CONFIG["checkpoint_file"],
            ollama_model=args.model,
            batch_write_size=CONFIG["batch_write_size"],
        )
