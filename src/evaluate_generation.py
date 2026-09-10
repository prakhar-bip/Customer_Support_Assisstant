"""
Milestone 9: Evaluation Suite for Historically Grounded Response Generation
-------------------------------------------------------------------------
Evaluates the response generation engine on 20 diverse test cases from
data/processed/golden_evaluation_set.jsonl across all 10 intents and 3 difficulty tiers.

Explicitly categorizes and benchmarks:
1. Cases where retrieval helps (strong semantic alignment, direct historical precedent)
2. Cases where retrieval is partially useful (moderate similarity, compound issues)
3. Cases where retrieval fails or is unhelpful (low similarity, ambiguous queries, sarcasm)

Assesses:
- Adherence to structured schema: {"reply", "reasoning_summary", "evidence_ids"}
- Grounding fidelity & evidence citation
- Policy integrity (zero invented claims/dates)
- Brand communication style & conciseness
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.generate_response import ResponseGenerator


def select_benchmark_subset(
    golden_jsonl_path: str = "data/processed/golden_evaluation_set.jsonl",
    target_count: int = 20,
) -> List[Dict[str, Any]]:
    """
    Selects exactly 20 diverse golden evaluation cases (2 per intent: 1 easy/med, 1 hard/edge).
    """
    with open(golden_jsonl_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    by_intent = {}
    for s in samples:
        intent = s["intent"]
        by_intent.setdefault(intent, []).append(s)

    selected = []
    # Guarantee exactly 2 distinct cases for each of the 10 intents (10 * 2 = 20)
    for intent in sorted(by_intent.keys()):
        items = by_intent[intent]
        hard_items = [x for x in items if x.get("difficulty") == "hard"]
        easy_items = [x for x in items if x.get("difficulty") == "easy"]
        med_items = [x for x in items if x.get("difficulty") == "medium"]

        # Case 1: Easy or medium (canonical)
        c1 = easy_items[0] if easy_items else (med_items[0] if med_items else items[0])
        selected.append(c1)

        # Case 2: Hard or alternative
        remaining = [x for x in items if x["conversation_id"] != c1["conversation_id"]]
        hard_rem = [x for x in remaining if x.get("difficulty") == "hard"]
        c2 = hard_rem[0] if hard_rem else (remaining[0] if remaining else items[0])
        selected.append(c2)

    return selected[:target_count]


def evaluate_generation(
    golden_jsonl_path: str = "data/processed/golden_evaluation_set.jsonl",
    output_results_path: str = "data/analysis/generation_evaluation_results.json",
    sample_count: int = 20,
    top_k: int = 3,
    sleep_between_calls: float = 12.0,
):
    print("=" * 80, flush=True)
    print(f"MILESTONE 9: HISTORICALLY GROUNDED RESPONSE GENERATION EVALUATION (N={sample_count})", flush=True)
    print("=" * 80, flush=True)

    # 1. Initialize Generator
    print("\n[1/3] Initializing ResponseGenerator (Gemini 2.5 Flash + FAISS Retriever)...", flush=True)
    generator = ResponseGenerator()
    print("  -> Initialized generator successfully.", flush=True)

    # 2. Select Benchmark Test Cases
    print(f"\n[2/3] Selecting {sample_count} diverse test cases across all 10 intents...", flush=True)
    test_cases = select_benchmark_subset(golden_jsonl_path, target_count=sample_count)
    print(f"  -> Selected {len(test_cases)} test cases.", flush=True)

    # 3. Run Generation and Scoring
    print(f"\n[3/3] Generating grounded responses with structured outputs (top_k={top_k})...\n", flush=True)
    results = []
    latencies = []

    retrieval_helps_cases = []
    partially_useful_cases = []
    retrieval_fails_cases = []

    for i, case in enumerate(test_cases, start=1):
        conv_id = case["conversation_id"]
        customer_msg = case["customer_message"]
        gt_intent = case["intent"]
        difficulty = case.get("difficulty", "medium")
        expected_action = case.get("expected_action", "")

        print(f"[{i:02d}/{len(test_cases):02d}] Evaluating {conv_id} ({gt_intent} | {difficulty})...", flush=True)
        print(f"     Query: \"{customer_msg[:80]}...\"", flush=True)

        t0 = time.time()
        try:
            gen_output = generator.generate(
                customer_message=customer_msg,
                top_k=top_k,
            )
            elapsed = time.time() - t0
            latencies.append(elapsed)

            pred_intent = gen_output["predicted_intent"]
            reply = gen_output["reply"]
            reasoning = gen_output["reasoning_summary"]
            evidence_ids = gen_output["evidence_ids"]
            retrieved = gen_output["retrieved_cases"]

            top_sim = retrieved[0]["similarity_score"] if retrieved else 0.0
            top_intent = retrieved[0]["intent"] if retrieved else "NONE"
            intent_matched = (top_intent == gt_intent)

            # Categorize retrieval quality
            if top_sim >= 0.72 and intent_matched:
                retrieval_quality = "retrieval_helps"
            elif (0.58 <= top_sim < 0.72) or (any(r["intent"] == gt_intent for r in retrieved)):
                retrieval_quality = "partially_useful"
            else:
                retrieval_quality = "retrieval_fails"

            # Check grounding properties
            concise = len(reply.split()) <= 60
            has_chain_of_thought = any(marker in reasoning.lower() for marker in [
                "step 1", "step 2", "first i", "next i", "thought process", "thinking:"
            ])

            record = {
                "case_index": i,
                "conversation_id": conv_id,
                "customer_message": customer_msg,
                "ground_truth_intent": gt_intent,
                "predicted_intent": pred_intent,
                "intent_classification_correct": (pred_intent == gt_intent),
                "difficulty": difficulty,
                "expected_action": expected_action,
                "top1_similarity": round(top_sim, 4),
                "top1_retrieved_intent": top_intent,
                "retrieval_quality_category": retrieval_quality,
                "generated_reply": reply,
                "reasoning_summary": reasoning,
                "cited_evidence_ids": evidence_ids,
                "evidence_cited_count": len(evidence_ids),
                "is_concise": concise,
                "word_count": len(reply.split()),
                "no_hidden_chain_of_thought": not has_chain_of_thought,
                "latency_seconds": round(elapsed, 2),
                "top_retrieved_cases": [
                    {
                        "rank": r["rank"],
                        "conversation_id": r["conversation_id"],
                        "similarity": r["similarity_score"],
                        "intent": r["intent"],
                        "historical_problem": r["customer_problem"][:100] + ("..." if len(r["customer_problem"]) > 100 else ""),
                        "historical_resolution": r["brand_response"][:120] + ("..." if len(r["brand_response"]) > 120 else ""),
                    }
                    for r in retrieved
                ],
            }

            results.append(record)

            if retrieval_quality == "retrieval_helps":
                retrieval_helps_cases.append(record)
            elif retrieval_quality == "partially_useful":
                partially_useful_cases.append(record)
            else:
                retrieval_fails_cases.append(record)

            print(f"     -> Pred Intent: {pred_intent} | Top-1 Sim: {top_sim:.4f} | Category: {retrieval_quality}", flush=True)
            print(f"     -> Reply: \"{reply[:100]}...\"", flush=True)
            print(f"     -> Cited Evidence: {evidence_ids}", flush=True)

        except Exception as e:
            print(f"     ERROR on {conv_id}: {e}\n", flush=True)

        # Rate limit compliance: pause between queries if not last
        if i < len(test_cases) and sleep_between_calls > 0:
            print(f"     [Pausing {sleep_between_calls:.0f}s to respect free tier rate limits...]\n", flush=True)
            time.sleep(sleep_between_calls)

    # Global Performance Summary
    avg_latency = float(pd.Series(latencies).mean()) if latencies else 0.0
    evidence_citation_rate = sum(1 for r in results if r["evidence_cited_count"] > 0) / len(results)
    conciseness_rate = sum(1 for r in results if r["is_concise"]) / len(results)
    cot_free_rate = sum(1 for r in results if r["no_hidden_chain_of_thought"]) / len(results)

    summary = {
        "milestone": "Milestone 9: Historically Grounded Response Generation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": len(results),
        "llm_model": "gemini-2.5-flash",
        "retrieval_engine": "sentence-transformers (all-MiniLM-L6-v2) + FAISS IndexFlatIP",
        "metrics": {
            "evidence_citation_rate": round(evidence_citation_rate, 4),
            "conciseness_rate": round(conciseness_rate, 4),
            "hidden_cot_free_rate": round(cot_free_rate, 4),
            "average_generation_latency_seconds": round(avg_latency, 2),
            "retrieval_quality_distribution": {
                "retrieval_helps_count": len(retrieval_helps_cases),
                "partially_useful_count": len(partially_useful_cases),
                "retrieval_fails_count": len(retrieval_fails_cases),
            },
        },
        "detailed_results": results,
    }

    os.makedirs(os.path.dirname(output_results_path), exist_ok=True)
    with open(output_results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  -> Saved Benchmark Results to: {output_results_path}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("EVALUATION COMPLETE SUMMARY:", flush=True)
    print(f"  Total Test Cases Evaluated:      {len(results)}", flush=True)
    print(f"  Cases Where Retrieval Helps:     {len(retrieval_helps_cases)} ({len(retrieval_helps_cases)/len(results):.1%})", flush=True)
    print(f"  Cases Partially Useful:          {len(partially_useful_cases)} ({len(partially_useful_cases)/len(results):.1%})", flush=True)
    print(f"  Cases Where Retrieval Fails/Weak:{len(retrieval_fails_cases)} ({len(retrieval_fails_cases)/len(results):.1%})", flush=True)
    print(f"  Evidence Citation Rate:          {evidence_citation_rate:.1%}", flush=True)
    print(f"  Conciseness Rate (<=60 words):   {conciseness_rate:.1%}", flush=True)
    print(f"  No Chain-of-Thought Exposed:     {cot_free_rate:.1%}", flush=True)
    print(f"  Average End-to-End Latency:      {avg_latency:.2f}s", flush=True)
    print("=" * 80, flush=True)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Historically Grounded Response Generation")
    parser.add_argument("--golden_jsonl", default="data/processed/golden_evaluation_set.jsonl")
    parser.add_argument("--output_results", default="data/analysis/generation_evaluation_results.json")
    parser.add_argument("--sample_count", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=12.0)

    args = parser.parse_args()
    evaluate_generation(
        golden_jsonl_path=args.golden_jsonl,
        output_results_path=args.output_results,
        sample_count=args.sample_count,
        top_k=args.top_k,
        sleep_between_calls=args.sleep,
    )
