"""
Conversation Reconstruction Pipeline for AmazonHelp
---------------------------------------------------
Milestone 3: Converts raw, fragmented tweet records into coherent,
multi-turn customer support conversations with a normalized schema.

Key Pipeline Stages:
1. Thread Graph Assembly: Connects parent-child tweet pointers using Disjoint Set Union (DSU).
2. Chronological Ordering: Sorts conversation events using ISO-8601 timestamps.
3. Role & Turn Identification: Differentiates customer queries from brand agent responses.
4. Noise Detection & Cleansing: Identifies non-English text, empty messages, and orphaned tweets.
5. Interaction Pair & History Structuring: Denormalizes pairs and tracks multi-turn trajectory.
6. Resolution Heuristic Classification: Categorizes threads as Resolved (Confirmed),
   Escalated (Private Channel), Guidance Provided, or Unresolved/Frustrated.
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct customer support conversations for a brand.")
    parser.add_argument(
        "--input-records",
        type=str,
        default="data/processed/amazonhelp_twcs_records.csv",
        help="Path to brand raw records CSV generated in Milestone 2."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save reconstructed conversation datasets."
    )
    parser.add_argument(
        "--brand",
        type=str,
        default="AmazonHelp",
        help="Target brand handle."
    )
    parser.add_argument(
        "--english-only",
        action="store_true",
        default=True,
        help="Filter out non-English / CJK text from the primary clean dataset."
    )
    return parser.parse_args()


# -------------------------------------------------------------
# Regex Rules & Heuristics
# -------------------------------------------------------------
RE_CJK = re.compile(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]')
RE_MENTIONS = re.compile(r'@\w+')
RE_URLS = re.compile(r'https?://\S+')
RE_HTML_ENTITIES = [
    (re.compile(r'&amp;'), '&'),
    (re.compile(r'&lt;'), '<'),
    (re.compile(r'&gt;'), '>'),
    (re.compile(r'&quot;'), '"'),
    (re.compile(r'&#39;'), "'"),
]

# Resolution & Sentiment Heuristics
RE_CUSTOMER_THANKS = re.compile(
    r'\b(thank|thanks|thx|thankyou|worked|fixed|sorted|resolved|appreciate|helpful|awesome|great now|got it)\b',
    re.IGNORECASE
)

RE_BRAND_ESCALATION = re.compile(
    r'\b(phone|chat|contact us|contact-us|reach us|fill out|form|unable to access|cannot access your account|security reasons|send us a dm|direct message)\b',
    re.IGNORECASE
)

RE_CUSTOMER_FRUSTRATION = re.compile(
    r'\b(still waiting|still haven\'t|not working|useless|terrible|awful|horrible|worst|disgusted|ridiculous|waste of time|boycott|cancel)\b',
    re.IGNORECASE
)


def clean_text(raw_text: str) -> str:
    """Clean HTML entities and normalize whitespace, preserving semantic words."""
    if not isinstance(raw_text, str):
        return ""
    text = raw_text
    for pattern, replacement in RE_HTML_ENTITIES:
        text = pattern.sub(replacement, text)
    return " ".join(text.split()).strip()


def strip_leading_mentions(text: str) -> str:
    """Remove leading @mentions from text while keeping internal mentions."""
    words = text.split()
    while words and words[0].startswith('@'):
        words.pop(0)
    return " ".join(words).strip()


def is_cjk_language(text: str) -> bool:
    """Returns True if text contains Chinese/Japanese/Korean characters."""
    return bool(RE_CJK.search(text))


def classify_resolution(
    turns: List[Dict[str, Any]]
) -> Tuple[str, float]:
    """
    Classifies the resolution status of a conversation thread.
    Returns: (resolution_status, confidence_score)
    
    Status Categories:
    - 'resolved_customer_confirmed': Customer explicitly confirmed fix/satisfaction.
    - 'escalated_private_channel': Brand requested handoff to phone/chat/form/DM for security/account access.
    - 'unresolved_frustrated': Customer expressed ongoing dissatisfaction/unresolved state at the end.
    - 'resolved_guidance_provided': Brand provided direct instructions/policy without negative rejoinder.
    - 'incomplete_or_abandoned': Only 1 turn or thread abruptly ended.
    """
    if len(turns) < 2:
        return ("incomplete_or_abandoned", 0.5)

    customer_turns = [t for t in turns if t["role"] == "customer"]
    brand_turns = [t for t in turns if t["role"] == "brand"]

    if not brand_turns:
        return ("unresolved_frustrated", 0.7)

    last_customer_text = customer_turns[-1]["text"] if customer_turns else ""
    last_brand_text = brand_turns[-1]["text"] if brand_turns else ""
    last_overall_turn = turns[-1]

    # Rule 1: Customer explicitly thanked or confirmed resolution
    if customer_turns and RE_CUSTOMER_THANKS.search(last_customer_text):
        return ("resolved_customer_confirmed", 0.95)

    # Rule 2: Escalation to private support channels
    all_brand_text = " ".join(t["text"] for t in brand_turns)
    if RE_BRAND_ESCALATION.search(all_brand_text):
        return ("escalated_private_channel", 0.90)

    # Rule 3: Customer ended thread with anger / still broken
    if last_overall_turn["role"] == "customer" and RE_CUSTOMER_FRUSTRATION.search(last_customer_text):
        return ("unresolved_frustrated", 0.85)

    # Rule 4: Brand provided guidance / troubleshooting, no further negative reply
    if last_overall_turn["role"] == "brand":
        return ("resolved_guidance_provided", 0.75)

    return ("unresolved_frustrated", 0.60)


def reconstruct_conversations(
    input_records_path: Path,
    output_dir: Path,
    brand: str,
    english_only: bool = True
) -> Dict[str, Any]:
    start_time = time.time()
    print(f"[*] Starting conversation reconstruction pipeline for brand: @{brand}")
    print(f"[*] Reading records from: {input_records_path}")

    if not input_records_path.exists():
        raise FileNotFoundError(f"Input records file not found at: {input_records_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_records_path, dtype={"author_id": str, "tweet_id": str})
    total_records = len(df)
    print(f"[+] Loaded {total_records:,} raw records.")

    # -------------------------------------------------------------
    # Step 1: Build Thread Graph using Disjoint Set Union (DSU)
    # -------------------------------------------------------------
    print("\n--- Step 1: Thread Graph Assembly via Disjoint Sets ---")
    parent_map = {}
    tweet_dict = {}

    for _, row in df.iterrows():
        tid = int(row["tweet_id"])
        pid = int(row["in_response_to_tweet_id"]) if pd.notna(row["in_response_to_tweet_id"]) else None
        parent_map[tid] = pid
        
        tweet_dict[tid] = {
            "tweet_id": tid,
            "author_id": str(row["author_id"]),
            "inbound": bool(row["inbound"]),
            "created_at": str(row["created_at"]),
            "text": str(row["text"]) if pd.notna(row["text"]) else "",
            "role": str(row["role"]) if "role" in row and pd.notna(row["role"]) else ("brand" if row["author_id"] == brand else "customer"),
            "parent_id": pid
        }

    # Union-Find
    parent_of = {tid: tid for tid in parent_map}

    def find(i):
        path = []
        while parent_of[i] != i:
            path.append(i)
            i = parent_of[i]
        for node in path:
            parent_of[node] = i
        return i

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent_of[root_i] = root_j

    resolved_edges = 0
    for tid, pid in parent_map.items():
        if pid is not None and pid in parent_of:
            union(tid, pid)
            resolved_edges += 1

    components = defaultdict(list)
    for tid in parent_of:
        root = find(tid)
        components[root].append(tid)

    total_threads = len(components)
    print(f"[+] Formed {total_threads:,} unique conversation components from {resolved_edges:,} linked edges.")

    # -------------------------------------------------------------
    # Step 2: Normalize & Process Conversation Components
    # -------------------------------------------------------------
    print("\n--- Step 2: Normalizing Conversation Components & Schema ---")
    clean_conversations = []
    noisy_conversations = []

    for root_id, tweet_ids in components.items():
        # Sort chronologically
        thread_tweets = [tweet_dict[tid] for tid in tweet_ids]
        thread_tweets.sort(
            key=lambda t: pd.to_datetime(t["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
        )

        # Basic thread properties
        turn_count = len(thread_tweets)
        first_tweet = thread_tweets[0]
        conversation_id = f"conv_{root_id}"
        
        # Determine root timestamp
        try:
            ts_dt = pd.to_datetime(first_tweet["created_at"], format="%a %b %d %H:%M:%S %z %Y")
            iso_timestamp = ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            iso_timestamp = first_tweet["created_at"]

        # Build turn list
        turns = []
        for idx, t in enumerate(thread_tweets):
            cleaned = clean_text(t["text"])
            turns.append({
                "turn": idx + 1,
                "tweet_id": t["tweet_id"],
                "role": t["role"],
                "author_id": t["author_id"],
                "created_at": t["created_at"],
                "text": cleaned,
                "text_raw": t["text"]
            })

        # Identify customer query and brand response
        customer_msgs = [t for t in turns if t["role"] == "customer"]
        brand_msgs = [t for t in turns if t["role"] == "brand"]

        primary_customer_msg = strip_leading_mentions(customer_msgs[0]["text"]) if customer_msgs else ""
        primary_brand_resp = brand_msgs[0]["text"] if brand_msgs else ""

        # Noise Checks
        is_noise = False
        noise_reasons = []

        # 1. Non-English / CJK check
        full_thread_text = " ".join(t["text"] for t in turns)
        if is_cjk_language(full_thread_text):
            is_noise = True
            noise_reasons.append("cjk_language")

        # 2. Missing customer or brand turn
        if not customer_msgs:
            is_noise = True
            noise_reasons.append("missing_customer_query")
        if not brand_msgs:
            is_noise = True
            noise_reasons.append("missing_brand_response")

        # 3. Degenerate text length
        if customer_msgs:
            if len(primary_customer_msg) < 5 and not customer_msgs[0]["text"].startswith("http"):
                is_noise = True
                noise_reasons.append("empty_or_degenerate_customer_message")

        # Resolution status
        res_status, res_conf = classify_resolution(turns)

        # Latency
        latency_sec = None
        if customer_msgs and brand_msgs:
            t_cust = pd.to_datetime(customer_msgs[0]["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
            t_brand = pd.to_datetime(brand_msgs[0]["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
            if pd.notna(t_cust) and pd.notna(t_brand):
                latency_sec = max(0.0, float((t_brand - t_cust).total_seconds()))

        conv_record = {
            "conversation_id": conversation_id,
            "timestamp": iso_timestamp,
            "customer_message": primary_customer_msg,
            "brand_response": primary_brand_resp,
            "conversation_history": turns,
            "metadata": {
                "turn_count": turn_count,
                "customer_author_id": customer_msgs[0]["author_id"] if customer_msgs else None,
                "brand_author_id": brand,
                "initial_response_latency_seconds": latency_sec,
                "initial_response_latency_minutes": round(latency_sec / 60.0, 2) if latency_sec is not None else None,
                "resolution_status": res_status,
                "resolution_confidence": res_conf,
                "has_url": "http" in full_thread_text,
                "is_multi_turn": turn_count > 2,
                "noise_reasons": noise_reasons
            }
        }

        if is_noise:
            noisy_conversations.append(conv_record)
        else:
            clean_conversations.append(conv_record)

    print(f"[+] Reconstructed {len(clean_conversations):,} clean conversations.")
    print(f"[+] Filtered {len(noisy_conversations):,} noisy/unusable conversations.")

    # -------------------------------------------------------------
    # Step 3: Export Clean Datasets
    # -------------------------------------------------------------
    print("\n--- Step 3: Exporting Normalized Schema Datasets ---")
    
    # 1. JSONL (Full nested structure)
    clean_jsonl_path = output_dir / f"{brand.lower()}_conversations_clean.jsonl"
    with open(clean_jsonl_path, "w", encoding="utf-8") as f:
        for c in clean_conversations:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[+] Saved clean JSONL to: {clean_jsonl_path}")

    # 2. Noisy JSONL
    noisy_jsonl_path = output_dir / f"{brand.lower()}_conversations_noisy.jsonl"
    with open(noisy_jsonl_path, "w", encoding="utf-8") as f:
        for c in noisy_conversations:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[+] Saved noisy JSONL to: {noisy_jsonl_path}")

    # 3. Tabular CSV representation for Pandas
    clean_flat = []
    for c in clean_conversations:
        clean_flat.append({
            "conversation_id": c["conversation_id"],
            "timestamp": c["timestamp"],
            "customer_message": c["customer_message"],
            "brand_response": c["brand_response"],
            "turn_count": c["metadata"]["turn_count"],
            "resolution_status": c["metadata"]["resolution_status"],
            "latency_minutes": c["metadata"]["initial_response_latency_minutes"],
            "has_url": c["metadata"]["has_url"],
            "conversation_history_json": json.dumps(c["conversation_history"], ensure_ascii=False)
        })
    df_clean = pd.DataFrame(clean_flat)
    clean_csv_path = output_dir / f"{brand.lower()}_conversations_clean.csv"
    df_clean.to_csv(clean_csv_path, index=False, encoding="utf-8")
    print(f"[+] Saved clean tabular CSV to: {clean_csv_path}")

    # -------------------------------------------------------------
    # Step 4: Generate Summary Metrics & Audit
    # -------------------------------------------------------------
    elapsed = time.time() - start_time
    resolution_counts = df_clean["resolution_status"].value_counts().to_dict()
    multi_turn_count = int((df_clean["turn_count"] > 2).sum())
    single_turn_count = int((df_clean["turn_count"] == 2).sum())

    summary_data = {
        "brand": brand,
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(elapsed, 2),
        "total_raw_records": total_records,
        "total_thread_components": total_threads,
        "clean_conversations_count": len(clean_conversations),
        "noisy_conversations_count": len(noisy_conversations),
        "clean_ratio_pct": round(len(clean_conversations) / total_threads * 100, 2),
        "single_turn_conversations_2_tweets": single_turn_count,
        "multi_turn_conversations_3_plus_tweets": multi_turn_count,
        "multi_turn_ratio_pct": round(multi_turn_count / len(clean_conversations) * 100, 2) if clean_conversations else 0,
        "median_latency_minutes": round(float(df_clean["latency_minutes"].median()), 2) if not df_clean.empty else 0,
        "resolution_distribution": {str(k): int(v) for k, v in resolution_counts.items()},
        "noise_reason_distribution": {str(k): int(v) for k, v in pd.Series([r for c in noisy_conversations for r in c["metadata"]["noise_reasons"]]).value_counts().items()},
        "output_files": [
            str(clean_jsonl_path.resolve()),
            str(clean_csv_path.resolve()),
            str(noisy_jsonl_path.resolve())
        ]
    }

    summary_path = output_dir / f"{brand.lower()}_conversation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[+] Saved conversation summary audit to: {summary_path}")
    print(f"\n[SUCCESS] Milestone 3 conversation reconstruction completed in {elapsed:.2f}s!\n")
    return summary_data


if __name__ == "__main__":
    args = parse_args()
    reconstruct_conversations(
        input_records_path=Path(args.input_records),
        output_dir=Path(args.output_dir),
        brand=args.brand,
        english_only=args.english_only
    )
