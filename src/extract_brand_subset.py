"""
Extract Brand Subset Pipeline
-----------------------------
Extracts a reproducible, high-integrity dataset subset for a single target brand
from the Kaggle Customer Support on Twitter (TWCS) dataset.

Features:
- Memory-efficient chunked streaming over 2.8M records.
- Bidirectional thread resolution: matches brand responses to customer parent queries.
- Exports both raw thread records and clean (Customer Query -> Brand Reply) pairs.
- Generates comprehensive metadata and integrity audit summary.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Extract reproducible brand subset from TWCS dataset.")
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/twcs/twcs.csv",
        help="Path to raw twcs.csv file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save the extracted subset files."
    )
    parser.add_argument(
        "--brand",
        type=str,
        default="AmazonHelp",
        help="Target brand handle (default: AmazonHelp)."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500000,
        help="Chunk size for streaming processing."
    )
    return parser.parse_args()


def extract_brand_subset(input_path: Path, output_dir: Path, brand: str, chunk_size: int = 500000):
    start_time = time.time()
    print(f"[*] Starting reproducible subset extraction for brand: @{brand}")
    print(f"[*] Input source: {input_path.resolve()}")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------
    # Pass 1: Extract all outbound brand tweets and target parent IDs
    # -------------------------------------------------------------
    print("\n--- Pass 1: Extracting outbound brand tweets ---")
    outbound_chunks = []
    parent_ids = set()
    total_rows_scanned = 0
    
    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        total_rows_scanned += len(chunk)
        brand_mask = (chunk["author_id"] == brand) & (chunk["inbound"] == False)
        brand_rows = chunk[brand_mask].copy()
        
        if not brand_rows.empty:
            outbound_chunks.append(brand_rows)
            valid_parents = brand_rows["in_response_to_tweet_id"].dropna().astype(np.int64)
            parent_ids.update(valid_parents)
            
        print(f"    Scanned {total_rows_scanned:,} rows | Found {len(parent_ids):,} unique parent inquiries targeted...")

    if not outbound_chunks:
        raise ValueError(f"No outbound tweets found for brand: {brand}")
        
    df_outbound = pd.concat(outbound_chunks, ignore_index=True)
    print(f"[+] Pass 1 Complete: {len(df_outbound):,} outbound tweets from @{brand}.")
    print(f"[+] Total parent tweet IDs to resolve: {len(parent_ids):,}")
    
    # -------------------------------------------------------------
    # Pass 2: Extract matching customer inbound parent tweets
    # -------------------------------------------------------------
    print("\n--- Pass 2: Resolving matching customer inquiries ---")
    inbound_chunks = []
    resolved_count = 0
    total_rows_scanned = 0
    
    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        total_rows_scanned += len(chunk)
        # Match tweets whose tweet_id is in parent_ids
        matched = chunk[chunk["tweet_id"].isin(parent_ids)].copy()
        if not matched.empty:
            inbound_chunks.append(matched)
            resolved_count += len(matched)
            
    df_inbound = pd.concat(inbound_chunks, ignore_index=True) if inbound_chunks else pd.DataFrame()
    match_rate = (len(df_inbound) / len(parent_ids) * 100) if parent_ids else 0.0
    print(f"[+] Pass 2 Complete: Successfully retrieved {len(df_inbound):,} customer parent tweets ({match_rate:.2f}% match rate).")

    # -------------------------------------------------------------
    # Step 3: Combine raw conversation events
    # -------------------------------------------------------------
    print("\n--- Step 3: Building consolidated brand dataset ---")
    df_inbound["role"] = "customer"
    df_outbound["role"] = "brand"
    
    df_combined = pd.concat([df_inbound, df_outbound], ignore_index=True)
    # Deduplicate in case a tweet was captured in both sets
    df_combined = df_combined.drop_duplicates(subset=["tweet_id"]).copy()
    
    # Sort chronologically
    df_combined["parsed_datetime"] = pd.to_datetime(
        df_combined["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce"
    )
    df_combined = df_combined.sort_values(by="parsed_datetime").reset_index(drop=True)
    
    raw_output_path = output_dir / f"{brand.lower()}_twcs_records.csv"
    df_combined.drop(columns=["parsed_datetime"]).to_csv(raw_output_path, index=False, encoding="utf-8")
    print(f"[+] Saved complete thread records to: {raw_output_path} ({len(df_combined):,} rows)")

    # -------------------------------------------------------------
    # Step 4: Construct Direct (Customer Query -> Brand Reply) Pairs
    # -------------------------------------------------------------
    print("\n--- Step 4: Generating structured interaction pairs ---")
    # Clean join keys
    df_outbound_replies = df_outbound[df_outbound["in_response_to_tweet_id"].notna()].copy()
    df_outbound_replies["in_response_to_tweet_id"] = df_outbound_replies["in_response_to_tweet_id"].astype(np.int64)
    
    pairs = pd.merge(
        df_inbound,
        df_outbound_replies,
        left_on="tweet_id",
        right_on="in_response_to_tweet_id",
        suffixes=("_customer", "_brand")
    )
    
    # Calculate response latency in seconds
    cust_time = pd.to_datetime(pairs["created_at_customer"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
    brand_time = pd.to_datetime(pairs["created_at_brand"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
    pairs["response_time_seconds"] = (brand_time - cust_time).dt.total_seconds()
    
    # Select and order key columns
    pair_columns = [
        "tweet_id_customer",
        "author_id_customer",
        "created_at_customer",
        "text_customer",
        "tweet_id_brand",
        "author_id_brand",
        "created_at_brand",
        "text_brand",
        "response_time_seconds"
    ]
    pairs_clean = pairs[pair_columns].copy()
    
    # Filter out any negative latency anomalies if clock drift exists
    pairs_clean = pairs_clean[pairs_clean["response_time_seconds"] >= 0].copy()
    
    pairs_output_path = output_dir / f"{brand.lower()}_pairs.csv"
    pairs_clean.to_csv(pairs_output_path, index=False, encoding="utf-8")
    print(f"[+] Saved structured pairs to: {pairs_output_path} ({len(pairs_clean):,} pairs)")

    # -------------------------------------------------------------
    # Step 5: Save Audit Metadata & Summary
    # -------------------------------------------------------------
    elapsed = time.time() - start_time
    summary_data = {
        "brand": brand,
        "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(elapsed, 2),
        "total_twcs_records_scanned": total_rows_scanned,
        "brand_outbound_tweets": len(df_outbound),
        "unique_customer_queries_targeted": len(parent_ids),
        "matched_customer_queries": len(df_inbound),
        "inquiry_match_rate_pct": round(match_rate, 2),
        "total_combined_records": len(df_combined),
        "total_structured_pairs": len(pairs_clean),
        "avg_response_time_minutes": round(float(pairs_clean["response_time_seconds"].median() / 60.0), 2) if len(pairs_clean) > 0 else 0,
        "files_generated": [
            str(raw_output_path.resolve()),
            str(pairs_output_path.resolve())
        ]
    }
    
    summary_path = output_dir / f"{brand.lower()}_subset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    print(f"[+] Saved dataset subset metadata to: {summary_path}")
    print(f"\n[SUCCESS] Milestone 2 brand subset extraction for @{brand} completed in {elapsed:.2f}s!\n")
    return summary_data


if __name__ == "__main__":
    args = parse_args()
    extract_brand_subset(
        input_path=Path(args.input_path),
        output_dir=Path(args.output_dir),
        brand=args.brand,
        chunk_size=args.chunk_size
    )
