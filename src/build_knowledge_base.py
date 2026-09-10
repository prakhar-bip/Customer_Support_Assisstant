"""
Milestone 8: Build Historical Resolution Knowledge Base & FAISS Index
---------------------------------------------------------------------
Extracts high-quality resolved customer support interactions from @AmazonHelp,
quarantines the 200 Golden Evaluation Set records, generates dense semantic embeddings
using sentence-transformers (all-MiniLM-L6-v2), and indexes them in FAISS IndexFlatIP.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import joblib
import faiss
from sentence_transformers import SentenceTransformer

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))


def format_dialogue_context(history_json_str: str) -> str:
    """Format full multi-turn conversation into readable dialogue context."""
    if not history_json_str or pd.isna(history_json_str):
        return ""
    try:
        turns = json.loads(history_json_str)
        formatted = []
        for t in turns:
            role = "Customer" if t.get("role") == "customer" else "Agent (@AmazonHelp)"
            turn_num = t.get("turn", len(formatted) + 1)
            text = t.get("text", "").strip()
            formatted.append(f"Turn {turn_num} [{role}]: {text}")
        return "\n".join(formatted)
    except Exception:
        return ""


def build_knowledge_base(
    clean_csv_path: str = "data/processed/amazonhelp_conversations_clean.csv",
    golden_jsonl_path: str = "data/processed/golden_evaluation_set.jsonl",
    intent_model_path: str = "models/intent_tfidf_logistic_regression.joblib",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    output_faiss_path: str = "models/historical_knowledge_base.faiss",
    output_metadata_path: str = "data/processed/historical_knowledge_base.jsonl",
    output_summary_path: str = "data/analysis/knowledge_base_summary.json",
    samples_per_class: int = 1200,
    batch_size: int = 128,
):
    print("=" * 80)
    print("MILESTONE 8: HISTORICAL KNOWLEDGE BASE CONSTRUCTION & FAISS INDEXING")
    print("=" * 80)
    start_total_time = time.time()

    # 1. Load Golden Set IDs for Anti-Leakage Quarantine
    print(f"\n[1/6] Loading Golden Evaluation Set from: {golden_jsonl_path}")
    if not os.path.exists(golden_jsonl_path):
        raise FileNotFoundError(f"Golden set not found: {golden_jsonl_path}")

    with open(golden_jsonl_path, "r", encoding="utf-8") as f:
        golden_ids = set(json.loads(line)["conversation_id"] for line in f)
    print(f"  -> Quarantining {len(golden_ids)} golden evaluation conversations.")

    # 2. Load Clean Conversations Corpus
    print(f"\n[2/6] Loading Clean Corpus from: {clean_csv_path}")
    if not os.path.exists(clean_csv_path):
        raise FileNotFoundError(f"Clean CSV not found: {clean_csv_path}")

    df = pd.read_csv(clean_csv_path)
    total_raw_clean = len(df)
    print(f"  -> Loaded {total_raw_clean:,} total clean conversations.")

    # 3. Filter for High-Quality Resolved Interactions
    print("\n[3/6] Applying Quality & Resolution Filters:")
    # Must have resolved status
    resolved_mask = df["resolution_status"].isin(
        ["resolved_guidance_provided", "resolved_customer_confirmed"]
    )
    # Anti-leakage quarantine
    quarantine_mask = ~df["conversation_id"].isin(golden_ids)
    # Quality text length filters
    length_mask = (
        (df["customer_message"].str.len().fillna(0) >= 15)
        & (df["brand_response"].str.len().fillna(0) >= 20)
    )

    eligible_mask = resolved_mask & quarantine_mask & length_mask
    df_eligible = df[eligible_mask].copy().reset_index(drop=True)
    print(f"  -> Eligible resolved candidates: {len(df_eligible):,}")

    # Double check anti-leakage
    overlap = set(df_eligible["conversation_id"]).intersection(golden_ids)
    assert len(overlap) == 0, f"FATAL: {len(overlap)} golden set IDs leaked into eligible pool!"
    print("  -> Anti-Leakage Verification: 0 overlapping records (PASSED).")

    # 4. Predict Intent & Confidence for Stratification
    print(f"\n[4/6] Classifying Intents using: {intent_model_path}")
    if not os.path.exists(intent_model_path):
        raise FileNotFoundError(f"Intent model not found: {intent_model_path}")

    intent_clf = joblib.load(intent_model_path)
    df_eligible["predicted_intent"] = intent_clf.predict(df_eligible["customer_message"])
    probs = intent_clf.predict_proba(df_eligible["customer_message"])
    df_eligible["intent_confidence"] = probs.max(axis=1)

    print("  -> Stratifying candidates across all 10 customer support intents...")
    kb_partitions = []
    class_distribution = {}

    for intent_code, group in df_eligible.groupby("predicted_intent"):
        sorted_group = group.sort_values("intent_confidence", ascending=False)
        selected = sorted_group.head(samples_per_class)
        kb_partitions.append(selected)
        class_distribution[intent_code] = len(selected)
        print(f"     - {intent_code:<40}: {len(selected):,} cases (mean conf: {selected['intent_confidence'].mean():.2%})")

    df_kb = pd.concat(kb_partitions, ignore_index=True)
    total_kb_records = len(df_kb)
    print(f"\n  -> Total Curated Knowledge Base Records: {total_kb_records:,}")

    # Format dialogue context
    print("  -> Formatting multi-turn dialogue histories...")
    df_kb["relevant_conversation_context"] = df_kb["conversation_history_json"].apply(format_dialogue_context)

    # 5. Generate Dense Semantic Embeddings using sentence-transformers
    print(f"\n[5/6] Encoding customer problems with '{embedding_model_name}' (batch_size={batch_size})...")
    embedder = SentenceTransformer(embedding_model_name)
    customer_texts = df_kb["customer_message"].tolist()

    t_enc_start = time.time()
    embeddings = embedder.encode(
        customer_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Crucial: L2 norm = 1 ensures inner product = cosine similarity
    )
    t_enc_elapsed = time.time() - t_enc_start
    print(f"  -> Encoded {len(customer_texts):,} texts in {t_enc_elapsed:.2f}s ({len(customer_texts)/t_enc_elapsed:.1f} texts/sec).")
    print(f"  -> Embeddings shape: {embeddings.shape} (dtype: {embeddings.dtype})")

    # Verify unit norm
    norms = np.linalg.norm(embeddings[:5], axis=1)
    print(f"  -> Verification of L2 normalization (sample norms): {np.round(norms, 4).tolist()}")

    # 6. Build FAISS IndexFlatIP
    print("\n[6/6] Constructing and Persisting FAISS Index & Metadata Store:")
    dim = embeddings.shape[1]
    # IndexFlatIP performs exact cosine similarity search on normalized vectors
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"  -> FAISS index created with {index.ntotal:,} vectors (dimension={dim}).")

    # Persist FAISS index
    os.makedirs(os.path.dirname(output_faiss_path), exist_ok=True)
    faiss.write_index(index, output_faiss_path)
    faiss_file_size_mb = os.path.getsize(output_faiss_path) / (1024 * 1024)
    print(f"  -> Saved FAISS index to: {output_faiss_path} ({faiss_file_size_mb:.2f} MB)")

    # Persist Metadata Store (JSONL)
    os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)
    with open(output_metadata_path, "w", encoding="utf-8") as f_out:
        for idx, row in df_kb.iterrows():
            record = {
                "kb_id": int(idx),
                "conversation_id": str(row["conversation_id"]),
                "customer_problem": str(row["customer_message"]),
                "relevant_conversation_context": str(row["relevant_conversation_context"]),
                "brand_response": str(row["brand_response"]),
                "intent": str(row["predicted_intent"]),
                "intent_confidence": round(float(row["intent_confidence"]), 4),
                "resolution_status": str(row["resolution_status"]),
                "timestamp": str(row.get("timestamp", "")),
                "turn_count": int(row.get("turn_count", 2)),
                "has_url": bool(row.get("has_url", False)),
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata_file_size_mb = os.path.getsize(output_metadata_path) / (1024 * 1024)
    print(f"  -> Saved Metadata Store to: {output_metadata_path} ({metadata_file_size_mb:.2f} MB)")

    # Export Audit Summary
    summary = {
        "milestone": "Milestone 8: Historical Resolution Retrieval",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": embedding_model_name,
        "embedding_dimension": dim,
        "faiss_index_type": "IndexFlatIP",
        "similarity_metric": "Cosine Similarity (via L2-normalized Inner Product)",
        "total_kb_records": total_kb_records,
        "golden_quarantined_count": len(golden_ids),
        "golden_overlap_count": 0,
        "samples_per_class_target": samples_per_class,
        "class_distribution": class_distribution,
        "mean_intent_confidence": round(float(df_kb["intent_confidence"].mean()), 4),
        "encoding_time_seconds": round(t_enc_elapsed, 2),
        "encoding_throughput_sentences_per_sec": round(len(customer_texts) / t_enc_elapsed, 1),
        "total_runtime_seconds": round(time.time() - start_total_time, 2),
        "faiss_file_path": output_faiss_path,
        "faiss_file_size_mb": round(faiss_file_size_mb, 2),
        "metadata_file_path": output_metadata_path,
        "metadata_file_size_mb": round(metadata_file_size_mb, 2),
    }

    os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)
    with open(output_summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)
    print(f"  -> Saved Summary Audit to: {output_summary_path}")

    print("\n" + "=" * 80)
    print(f"KNOWLEDGE BASE CONSTRUCTION COMPLETE IN {time.time() - start_total_time:.2f}s!")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Historical Knowledge Base & FAISS Index")
    parser.add_argument("--clean_csv", default="data/processed/amazonhelp_conversations_clean.csv")
    parser.add_argument("--golden_jsonl", default="data/processed/golden_evaluation_set.jsonl")
    parser.add_argument("--intent_model", default="models/intent_tfidf_logistic_regression.joblib")
    parser.add_argument("--output_faiss", default="models/historical_knowledge_base.faiss")
    parser.add_argument("--output_metadata", default="data/processed/historical_knowledge_base.jsonl")
    parser.add_argument("--output_summary", default="data/analysis/knowledge_base_summary.json")
    parser.add_argument("--samples_per_class", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()
    build_knowledge_base(
        clean_csv_path=args.clean_csv,
        golden_jsonl_path=args.golden_jsonl,
        intent_model_path=args.intent_model,
        output_faiss_path=args.output_faiss,
        output_metadata_path=args.output_metadata,
        output_summary_path=args.output_summary,
        samples_per_class=args.samples_per_class,
        batch_size=args.batch_size,
    )
