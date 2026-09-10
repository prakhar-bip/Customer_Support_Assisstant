"""
Escalation Threshold Tuning & Calibration Engine (@AmazonHelp).

Calibrates the decision thresholds for the multi-signal escalation policy
using ONLY held-out validation data from the clean training pool.

CRITICAL INTEGRITY CONSTRAINT:
  - Zero data leakage from the Golden Evaluation Set (data/processed/golden_evaluation_set.jsonl).
  - The 200 golden evaluation records are strictly quarantined.
  - Thresholds are tuned on a held-out validation set of 500 clean conversation examples.

Design Philosophy:
  - Optimize for TRUSTWORTHY AUTOMATION over maximum automation.
  - Primary goal: Auto-Handle Precision >= 90% (when the AI resolves a ticket, it must be safe).
  - Secondary goal: Escalation Recall >= 85% (catch at least 85% of complex/high-risk cases).
  - Maximize the Automation Rate subject to these safety constraints.

Exports results to:
  data/analysis/escalation_threshold_calibration.json
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.predict_intent import predict_intent
from src.retrieve_resolutions import ResolutionRetriever
from src.escalation_policy import (
    extract_decision_signals,
    make_routing_decision,
    SAFETY_PATTERNS,
    MANDATORY_ESCALATION_INTENTS
)

CLEAN_DATA_PATH = "data/processed/amazonhelp_conversations_clean.csv"
GOLDEN_SET_PATH = "data/processed/golden_evaluation_set.jsonl"
CALIBRATION_OUTPUT_PATH = "data/analysis/escalation_threshold_calibration.json"

TAXONOMY_INTENTS = [
    "ORDER_DELIVERY_AND_TRACKING",
    "REFUND_STATUS_AND_DISPUTES",
    "DAMAGED_DEFECTIVE_OR_WRONG_ITEM",
    "PRIME_MEMBERSHIP_AND_DIGITAL",
    "ORDER_CANCELLATION",
    "PAYMENT_BILLING_AND_PROMOS",
    "CUSTOMER_SERVICE_AND_COURIER_FEEDBACK",
    "ACCOUNT_ACCESS_AND_SECURITY",
    "TECHNICAL_AND_PLATFORM_ISSUES",
    "RETURNS_AND_EXCHANGES"
]

def load_quarantined_validation_pool(sample_size_per_intent: int = 50, random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Sample a stratified validation pool from the training set, strictly
    excluding all golden evaluation conversations.
    """
    print("=" * 80)
    print("STEP 1: CURATING QUARANTINED VALIDATION POOL (ZERO LEAKAGE)")
    print("=" * 80)

    # 1. Load Golden Set IDs
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_records = [json.loads(line) for line in f if line.strip()]
    golden_ids = set(r["conversation_id"] for r in golden_records)
    print(f"Isolated {len(golden_ids)} Golden Evaluation Set conversation IDs.")

    # 2. Load Clean Conversations
    df_clean = pd.read_csv(CLEAN_DATA_PATH)
    total_clean = len(df_clean)
    print(f"Loaded {total_clean:,} total clean conversations from {CLEAN_DATA_PATH}.")

    # Exclude Golden Set
    df_pool = df_clean[~df_clean["conversation_id"].isin(golden_ids)].copy()
    overlap = set(df_pool["conversation_id"]).intersection(golden_ids)
    assert len(overlap) == 0, f"FATAL ERROR: Golden set leakage detected! {len(overlap)} IDs overlap."
    print(f"Quarantined training pool: {len(df_pool):,} rows (Zero Golden Set overlap verified).")

    # 3. Assign Intent and Escalation Ground Truth to Validation Candidates
    from src.discover_intents import TAXONOMY_PATTERNS
    compiled_patterns = {k: re.compile(v, re.I) for k, v in TAXONOMY_PATTERNS.items()}

    precedence = [
        "DAMAGED_DEFECTIVE_OR_WRONG_ITEM",
        "ORDER_CANCELLATION",
        "RETURNS_AND_EXCHANGES",
        "REFUND_STATUS_AND_DISPUTES",
        "PAYMENT_BILLING_AND_PROMOS",
        "ORDER_DELIVERY_AND_TRACKING",
        "PRIME_MEMBERSHIP_AND_DIGITAL",
        "ACCOUNT_ACCESS_AND_SECURITY",
        "TECHNICAL_AND_PLATFORM_ISSUES",
        "CUSTOMER_SERVICE_AND_COURIER_FEEDBACK"
    ]

    print("\nAssigning intents and ground truth escalation labels to pool...")
    assigned_intents = []
    for msg in df_pool["customer_message"].fillna("").astype(str):
        hits = [k for k, p in compiled_patterns.items() if p.search(msg)]
        if len(hits) == 1:
            assigned_intents.append(hits[0])
        elif len(hits) > 1:
            assigned_intents.append(next((p for p in precedence if p in hits), hits[0]))
        else:
            assigned_intents.append(None)

    df_pool["assigned_intent"] = assigned_intents
    df_labeled = df_pool.dropna(subset=["assigned_intent"]).copy()

    # Sample stratified validation cases
    validation_records = []
    for intent in TAXONOMY_INTENTS:
        sub_df = df_labeled[df_labeled["assigned_intent"] == intent]
        sampled = sub_df.sample(n=min(sample_size_per_intent, len(sub_df)), random_state=random_state)
        for _, row in sampled.iterrows():
            cid = str(row["conversation_id"])
            msg = str(row["customer_message"]).strip()
            turn_count = int(row.get("turn_count", 2))
            res_status = str(row.get("resolution_status", "resolved_guidance_provided"))

            # Escalation Ground Truth (Identical rubric to LABELING_GUIDE.md and Milestone 5)
            if intent in ["ACCOUNT_ACCESS_AND_SECURITY", "CUSTOMER_SERVICE_AND_COURIER_FEEDBACK"]:
                gt_decision = "ESCALATE"
                gt_reason = f"Mandatory human escalation for {intent}."
            elif res_status == "escalated_private_channel" or turn_count >= 6:
                gt_decision = "ESCALATE"
                gt_reason = "Required multi-turn escalation or private channel resolution."
            elif intent == "REFUND_STATUS_AND_DISPUTES" and re.search(r'\b(a-?to-?z|claim|months?|weeks?|stolen|fraud|bank)\b', msg, re.I):
                gt_decision = "ESCALATE"
                gt_reason = "Complex financial dispute requiring transaction lookup."
            elif intent == "DAMAGED_DEFECTIVE_OR_WRONG_ITEM" and re.search(r'\b(tampered|counterfeit|fake|wrong item|dead)\b', msg, re.I):
                gt_decision = "ESCALATE"
                gt_reason = "Product authenticity or condition dispute requiring merchant investigation."
            elif intent == "ORDER_DELIVERY_AND_TRACKING" and re.search(r'\b(marked\s+delivered|says?\s+delivered|showing\s+delivered)\b', msg, re.I):
                gt_decision = "ESCALATE"
                gt_reason = "False delivery dispute requiring carrier trace."
            else:
                gt_decision = "AUTO_HANDLE"
                gt_reason = "Standard self-service query or FAQ resolvable via policy/links."

            validation_records.append({
                "conversation_id": cid,
                "customer_message": msg,
                "ground_truth_intent": intent,
                "turn_count": turn_count,
                "ground_truth_decision": gt_decision,
                "ground_truth_reason": gt_reason
            })

    print(f"Successfully sampled {len(validation_records)} validation records.")
    auto_cnt = sum(1 for r in validation_records if r["ground_truth_decision"] == "AUTO_HANDLE")
    esc_cnt = sum(1 for r in validation_records if r["ground_truth_decision"] == "ESCALATE")
    print(f"Validation Distribution: AUTO_HANDLE = {auto_cnt} ({auto_cnt/len(validation_records)*100:.1f}%), ESCALATE = {esc_cnt} ({esc_cnt/len(validation_records)*100:.1f}%)")

    return validation_records

def extract_validation_signals(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract measurable signals for all validation records using cached retriever."""
    print("\n" + "=" * 80)
    print("STEP 2: EXTRACTING MEASURABLE SIGNALS FOR VALIDATION SAMPLES")
    print("=" * 80)

    retriever = ResolutionRetriever()
    enriched_records = []

    for rec in tqdm(records, desc="Extracting Signals"):
        msg = rec["customer_message"]
        signals = extract_decision_signals(
            text=msg,
            top_k=5,
            evidence_sim_threshold=0.65,
            retriever=retriever
        )
        rec_copy = rec.copy()
        rec_copy["signals"] = signals
        enriched_records.append(rec_copy)

    return enriched_records

def evaluate_threshold_configuration(
    records: List[Dict[str, Any]],
    thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """Evaluate decision policy with given thresholds on precomputed signals."""
    tp_auto = 0  # True Auto-Handle
    fp_auto = 0  # False Auto-Handle (CRITICAL FAILURE: predicted AUTO_HANDLE when true is ESCALATE)
    tp_esc = 0   # True Escalate
    fp_esc = 0   # False Escalate (Safe conservative error: predicted ESCALATE when true is AUTO_HANDLE)

    for rec in records:
        gt = rec["ground_truth_decision"]
        res = make_routing_decision(
            text=rec["customer_message"],
            thresholds=thresholds,
            signals=rec["signals"]
        )
        pred = res["decision"]

        if pred == "AUTO_HANDLE" and gt == "AUTO_HANDLE":
            tp_auto += 1
        elif pred == "AUTO_HANDLE" and gt == "ESCALATE":
            fp_auto += 1
        elif pred == "ESCALATE" and gt == "ESCALATE":
            tp_esc += 1
        elif pred == "ESCALATE" and gt == "AUTO_HANDLE":
            fp_esc += 1

    total = len(records)
    total_pred_auto = tp_auto + fp_auto
    total_pred_esc = tp_esc + fp_esc
    total_true_auto = tp_auto + fp_esc
    total_true_esc = tp_esc + fp_auto

    automation_rate = round(total_pred_auto / total, 4) if total > 0 else 0.0
    auto_precision = round(tp_auto / total_pred_auto, 4) if total_pred_auto > 0 else 0.0
    auto_recall = round(tp_auto / total_true_auto, 4) if total_true_auto > 0 else 0.0
    esc_precision = round(tp_esc / total_pred_esc, 4) if total_pred_esc > 0 else 0.0
    esc_recall = round(tp_esc / total_true_esc, 4) if total_true_esc > 0 else 0.0
    accuracy = round((tp_auto + tp_esc) / total, 4)

    return {
        "thresholds": thresholds,
        "accuracy": accuracy,
        "automation_rate": automation_rate,
        "auto_handle_precision": auto_precision,
        "auto_handle_recall": auto_recall,
        "escalation_precision": esc_precision,
        "escalation_recall": esc_recall,
        "false_auto_handle_count": fp_auto,
        "false_auto_handle_rate": round(fp_auto / total_pred_auto, 4) if total_pred_auto > 0 else 0.0,
        "confusion_matrix": {
            "true_auto_handle": tp_auto,
            "false_auto_handle": fp_auto,
            "true_escalate": tp_esc,
            "false_escalate": fp_esc
        }
    }

def run_grid_calibration(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Search over candidate threshold grids and select the optimal Trustworthy
    Operating Point.
    """
    print("\n" + "=" * 80)
    print("STEP 3: RUNNING GRID CALIBRATION OVER THRESHOLD CANDIDATES")
    print("=" * 80)

    # Candidate threshold grids
    conf_grid = [0.40, 0.50, 0.60, 0.70]
    margin_grid = [0.05, 0.10, 0.15, 0.20]
    retrieval_grid = [0.55, 0.60, 0.65, 0.70, 0.75]
    evidence_grid = [1, 2]

    all_results = []

    print(f"Sweeping {len(conf_grid) * len(margin_grid) * len(retrieval_grid) * len(evidence_grid)} configurations...")
    for conf in conf_grid:
        for margin in margin_grid:
            for sim in retrieval_grid:
                for ev in evidence_grid:
                    t = {
                        "min_intent_confidence": conf,
                        "min_intent_margin": margin,
                        "min_retrieval_score": sim,
                        "min_evidence_count": ev,
                        "evidence_similarity_threshold": 0.65
                    }
                    eval_res = evaluate_threshold_configuration(records, t)
                    all_results.append(eval_res)

    # Filter by Trustworthy Automation Constraints:
    # 1. Auto-Handle Precision >= 0.90 (At least 90% of automated replies are safe)
    # 2. Escalation Recall >= 0.85 (Catches >= 85% of complex human situations)
    valid_candidates = [
        r for r in all_results
        if r["auto_handle_precision"] >= 0.90 and r["escalation_recall"] >= 0.85
    ]

    print(f"\nFound {len(valid_candidates)} configurations meeting Trustworthy Automation constraints:")
    print("  Constraint 1: Auto-Handle Precision >= 90.0%")
    print("  Constraint 2: Escalation Recall >= 85.0%")

    if valid_candidates:
        # Select configuration that maximizes Automation Rate among trustworthy candidates
        best_cfg = max(valid_candidates, key=lambda x: (x["automation_rate"], x["accuracy"]))
    else:
        print("Warning: No candidate met strict constraints. Selecting configuration with highest Auto-Handle Precision.")
        best_cfg = max(all_results, key=lambda x: (x["auto_handle_precision"], x["escalation_recall"]))

    print("\n" + "=" * 80)
    print("SELECTED TRUSTWORTHY OPERATING POINT (CALIBRATED ON VALIDATION DATA):")
    print("=" * 80)
    t = best_cfg["thresholds"]
    print(f"  min_intent_confidence:         {t['min_intent_confidence']}")
    print(f"  min_intent_margin:             {t['min_intent_margin']}")
    print(f"  min_retrieval_score:           {t['min_retrieval_score']}")
    print(f"  min_evidence_count:            {t['min_evidence_count']}")
    print(f"  evidence_similarity_threshold: {t['evidence_similarity_threshold']}")
    print("-" * 80)
    print(f"  Validation Accuracy:           {best_cfg['accuracy'] * 100:.2f}%")
    print(f"  Validation Automation Rate:    {best_cfg['automation_rate'] * 100:.2f}%")
    print(f"  Auto-Handle Precision:         {best_cfg['auto_handle_precision'] * 100:.2f}%")
    print(f"  Auto-Handle Recall:            {best_cfg['auto_handle_recall'] * 100:.2f}%")
    print(f"  Escalation Precision:          {best_cfg['escalation_precision'] * 100:.2f}%")
    print(f"  Escalation Recall:             {best_cfg['escalation_recall'] * 100:.2f}%")
    print(f"  False Auto-Handle Rate:        {best_cfg['false_auto_handle_rate'] * 100:.2f}% (Safety Critical)")
    print("=" * 80)

    # Sort grid by automation rate for tradeoff analysis
    tradeoff_curve = sorted(all_results, key=lambda x: x["automation_rate"])

    output_payload = {
        "milestone": "Milestone 10: Auto-Handle vs Human Escalation",
        "objective": "Trustworthy Automation Threshold Calibration",
        "validation_sample_count": len(records),
        "selected_trustworthy_thresholds": best_cfg["thresholds"],
        "validation_performance": {
            "accuracy": best_cfg["accuracy"],
            "automation_rate": best_cfg["automation_rate"],
            "auto_handle_precision": best_cfg["auto_handle_precision"],
            "auto_handle_recall": best_cfg["auto_handle_recall"],
            "escalation_precision": best_cfg["escalation_precision"],
            "escalation_recall": best_cfg["escalation_recall"],
            "false_auto_handle_count": best_cfg["false_auto_handle_count"],
            "false_auto_handle_rate": best_cfg["false_auto_handle_rate"],
            "confusion_matrix": best_cfg["confusion_matrix"]
        },
        "sample_tradeoff_grid": [
            {
                "automation_rate": r["automation_rate"],
                "auto_handle_precision": r["auto_handle_precision"],
                "escalation_recall": r["escalation_recall"],
                "thresholds": r["thresholds"]
            }
            for r in tradeoff_curve[::20]  # Subsample for reporting
        ]
    }

    os.makedirs(os.path.dirname(CALIBRATION_OUTPUT_PATH), exist_ok=True)
    with open(CALIBRATION_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[SUCCESS] Calibration saved to: {CALIBRATION_OUTPUT_PATH}")
    return output_payload

def main():
    validation_records = load_quarantined_validation_pool(sample_size_per_intent=50)
    enriched_records = extract_validation_signals(validation_records)
    run_grid_calibration(enriched_records)

if __name__ == "__main__":
    main()
