"""
Milestone 8: Retrieval Quality Evaluation on Golden Benchmark
-------------------------------------------------------------
Evaluates the FAISS + sentence-transformers retrieval system blindly against
the 200 hand-reviewed cases in data/processed/golden_evaluation_set.jsonl.

Computes:
- Hit@1, Hit@3, Hit@5 (Intent Alignment)
- Mean Reciprocal Rank (MRR)
- Mean Average Precision @ 5 (MAP@5)
- Mean Cosine Similarity (Overall & by Difficulty: Easy, Medium, Hard)
- Per-Intent Retrieval Metrics
- Detailed Case Studies (Successful, Partially Useful, and Bad/Misleading)
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.retrieve_resolutions import ResolutionRetriever


def evaluate_retrieval(
    golden_jsonl_path: str = "data/processed/golden_evaluation_set.jsonl",
    faiss_index_path: str = "models/historical_knowledge_base.faiss",
    metadata_path: str = "data/processed/historical_knowledge_base.jsonl",
    output_metrics_path: str = "data/analysis/retrieval_evaluation_metrics.json",
    top_k: int = 5,
):
    print("=" * 80)
    print("MILESTONE 8: HISTORICAL RETRIEVAL EVALUATION ON GOLDEN SET")
    print("=" * 80)

    # 1. Initialize Retriever
    print(f"\n[1/4] Initializing ResolutionRetriever...")
    t0 = time.time()
    retriever = ResolutionRetriever(
        faiss_index_path=faiss_index_path,
        metadata_path=metadata_path,
    )
    print(f"  -> Loaded {retriever.total_vectors:,} historical cases in {time.time() - t0:.2f}s.")

    # 2. Load Golden Evaluation Set
    print(f"\n[2/4] Loading Golden Benchmark from: {golden_jsonl_path}")
    golden_samples = []
    with open(golden_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                golden_samples.append(json.loads(line))
    total_queries = len(golden_samples)
    print(f"  -> Loaded {total_queries} quarantined golden evaluation samples.")

    # 3. Run Retrieval and Scoring
    print(f"\n[3/4] Evaluating Retrieval across {total_queries} queries (top_k={top_k})...")
    eval_records = []
    latencies = []

    hit_1_list = []
    hit_3_list = []
    hit_5_list = []
    reciprocal_ranks = []
    ap_5_list = []
    top1_similarities = []
    top5_mean_similarities = []

    for idx, sample in enumerate(golden_samples):
        query = sample["customer_message"]
        gt_intent = sample["intent"]
        gt_action = sample.get("expected_action", "")
        difficulty = sample.get("difficulty", "medium")
        conv_id = sample["conversation_id"]

        q_start = time.time()
        retrieved = retriever.retrieve(query=query, top_k=top_k)
        latencies.append(time.time() - q_start)

        # Evaluate matches
        intent_matches = [res["intent"] == gt_intent for res in retrieved]
        sims = [res["similarity_score"] for res in retrieved]

        hit_1 = intent_matches[0] if len(intent_matches) > 0 else False
        hit_3 = any(intent_matches[:3]) if len(intent_matches) > 0 else False
        hit_5 = any(intent_matches[:5]) if len(intent_matches) > 0 else False

        # Reciprocal Rank
        rr = 0.0
        for rank_idx, match in enumerate(intent_matches, start=1):
            if match:
                rr = 1.0 / rank_idx
                break

        # Average Precision @ 5
        num_relevant = sum(intent_matches)
        if num_relevant > 0:
            precisions = []
            running_rel = 0
            for rank_idx, match in enumerate(intent_matches, start=1):
                if match:
                    running_rel += 1
                    precisions.append(running_rel / rank_idx)
            ap_5 = sum(precisions) / num_relevant
        else:
            ap_5 = 0.0

        hit_1_list.append(1 if hit_1 else 0)
        hit_3_list.append(1 if hit_3 else 0)
        hit_5_list.append(1 if hit_5 else 0)
        reciprocal_ranks.append(rr)
        ap_5_list.append(ap_5)
        top1_similarities.append(sims[0] if sims else 0.0)
        top5_mean_similarities.append(float(np.mean(sims)) if sims else 0.0)

        eval_records.append({
            "golden_conv_id": conv_id,
            "query": query,
            "ground_truth_intent": gt_intent,
            "expected_action": gt_action,
            "difficulty": difficulty,
            "hit_1": hit_1,
            "hit_3": hit_3,
            "hit_5": hit_5,
            "reciprocal_rank": round(rr, 4),
            "ap_5": round(ap_5, 4),
            "top1_similarity": round(sims[0], 4) if sims else 0.0,
            "top5_mean_similarity": round(float(np.mean(sims)), 4) if sims else 0.0,
            "retrieved_cases": retrieved,
        })

    # Global Aggregates
    macro_hit_1 = float(np.mean(hit_1_list))
    macro_hit_3 = float(np.mean(hit_3_list))
    macro_hit_5 = float(np.mean(hit_5_list))
    mrr = float(np.mean(reciprocal_ranks))
    map_5 = float(np.mean(ap_5_list))
    mean_top1_sim = float(np.mean(top1_similarities))
    mean_top5_sim = float(np.mean(top5_mean_similarities))
    avg_latency_ms = float(np.mean(latencies)) * 1000

    print(f"\n--- GLOBAL RETRIEVAL METRICS (N = {total_queries}) ---")
    print(f"  Top-1 Intent Hit Rate (Precision@1): {macro_hit_1:.2%}")
    print(f"  Top-3 Intent Hit Rate (Hit@3):        {macro_hit_3:.2%}")
    print(f"  Top-5 Intent Hit Rate (Hit@5):        {macro_hit_5:.2%}")
    print(f"  Mean Reciprocal Rank (MRR):           {mrr:.4f}")
    print(f"  Mean Average Precision @ 5 (MAP@5):   {map_5:.4f}")
    print(f"  Mean Top-1 Cosine Similarity:         {mean_top1_sim:.4f}")
    print(f"  Mean Top-5 Cosine Similarity:         {mean_top5_sim:.4f}")
    print(f"  Average Query Latency:                {avg_latency_ms:.2f} ms")

    # Metrics by Difficulty
    df_eval = pd.DataFrame(eval_records)
    difficulty_metrics = {}
    print("\n--- PERFORMANCE BY DIFFICULTY TIER ---")
    for diff in ["easy", "medium", "hard"]:
        sub = df_eval[df_eval["difficulty"] == diff]
        if len(sub) > 0:
            diff_hit1 = float(sub["hit_1"].mean())
            diff_hit3 = float(sub["hit_3"].mean())
            diff_hit5 = float(sub["hit_5"].mean())
            diff_mrr = float(sub["reciprocal_rank"].mean())
            diff_sim = float(sub["top1_similarity"].mean())
            difficulty_metrics[diff] = {
                "count": len(sub),
                "hit_1": round(diff_hit1, 4),
                "hit_3": round(diff_hit3, 4),
                "hit_5": round(diff_hit5, 4),
                "mrr": round(diff_mrr, 4),
                "mean_top1_similarity": round(diff_sim, 4),
            }
            print(f"  Tier: {diff.upper():<6} (N={len(sub):2d}) | Hit@1: {diff_hit1:6.2%} | Hit@3: {diff_hit3:6.2%} | Hit@5: {diff_hit5:6.2%} | MRR: {diff_mrr:.4f} | Sim: {diff_sim:.4f}")

    # Metrics by Intent
    intent_metrics = {}
    print("\n--- PERFORMANCE BY INTENT CLASS ---")
    for intent_code, sub in df_eval.groupby("ground_truth_intent"):
        int_hit1 = float(sub["hit_1"].mean())
        int_hit3 = float(sub["hit_3"].mean())
        int_hit5 = float(sub["hit_5"].mean())
        int_mrr = float(sub["reciprocal_rank"].mean())
        int_sim = float(sub["top1_similarity"].mean())
        intent_metrics[intent_code] = {
            "count": len(sub),
            "hit_1": round(int_hit1, 4),
            "hit_3": round(int_hit3, 4),
            "hit_5": round(int_hit5, 4),
            "mrr": round(int_mrr, 4),
            "mean_top1_similarity": round(int_sim, 4),
        }
        print(f"  {intent_code:<40} (N={len(sub):2d}) | Hit@1: {int_hit1:6.2%} | Hit@3: {int_hit3:6.2%} | Hit@5: {int_hit5:6.2%} | MRR: {int_mrr:.4f}")

    # 4. Extract Real Case Studies (Successful, Partially Useful, Bad)
    print("\n[4/4] Extracting Qualitative Case Studies...")
    # Successful: High similarity (>0.82), hit_1 is True, clear actionable brand resolution
    successful_candidates = [
        r for r in eval_records
        if r["hit_1"] and r["top1_similarity"] >= 0.82 and len(r["retrieved_cases"][0]["brand_response"]) >= 40
    ]
    successful_cases = sorted(successful_candidates, key=lambda x: x["top1_similarity"], reverse=True)[:3]

    # Partially Useful: Moderate similarity (0.65 - 0.76), hit_1 is True or hit_3 is True
    partially_candidates = [
        r for r in eval_records
        if (0.65 <= r["top1_similarity"] <= 0.77) and r["hit_3"]
    ]
    partially_useful_cases = sorted(partially_candidates, key=lambda x: x["top1_similarity"])[:3]

    # Bad / Misleading: hit_1 is False and hit_5 is False (or top1_similarity is low or wrong intent)
    bad_candidates = [
        r for r in eval_records
        if not r["hit_1"] and not r["hit_3"]
    ]
    if not bad_candidates:
        bad_candidates = [r for r in eval_records if not r["hit_1"]]
    bad_cases = sorted(bad_candidates, key=lambda x: x["top1_similarity"])[:3]

    # Build final report dictionary
    report = {
        "milestone": "Milestone 8: Historical Resolution Retrieval",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": total_queries,
        "knowledge_base_size": retriever.total_vectors,
        "global_metrics": {
            "hit_at_1": round(macro_hit_1, 4),
            "hit_at_3": round(macro_hit_3, 4),
            "hit_at_5": round(macro_hit_5, 4),
            "mean_reciprocal_rank_mrr": round(mrr, 4),
            "mean_average_precision_map5": round(map_5, 4),
            "mean_top1_similarity": round(mean_top1_sim, 4),
            "mean_top5_similarity": round(mean_top5_sim, 4),
            "average_query_latency_ms": round(avg_latency_ms, 2),
        },
        "difficulty_breakdown": difficulty_metrics,
        "intent_breakdown": intent_metrics,
        "case_studies": {
            "successful_retrievals": successful_cases,
            "partially_useful_retrievals": partially_useful_cases,
            "bad_retrievals": bad_cases,
        },
    }

    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
    with open(output_metrics_path, "w", encoding="utf-8") as f_out:
        json.dump(report, f_out, indent=2, ensure_ascii=False)
    print(f"\n  -> Saved Evaluation Metrics & Case Studies to: {output_metrics_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Historical Retrieval Engine")
    parser.add_argument("--golden_jsonl", default="data/processed/golden_evaluation_set.jsonl")
    parser.add_argument("--faiss_index", default="models/historical_knowledge_base.faiss")
    parser.add_argument("--metadata", default="data/processed/historical_knowledge_base.jsonl")
    parser.add_argument("--output_metrics", default="data/analysis/retrieval_evaluation_metrics.json")
    parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()
    evaluate_retrieval(
        golden_jsonl_path=args.golden_jsonl,
        faiss_index_path=args.faiss_index,
        metadata_path=args.metadata,
        output_metrics_path=args.output_metrics,
        top_k=args.top_k,
    )
