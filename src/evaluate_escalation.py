"""
Evaluation of Escalation Policy on Golden Evaluation Set (N = 200).

Applies the frozen multi-signal routing policy (calibrated on validation data)
to the untouched Golden Evaluation Set (data/processed/golden_evaluation_set.jsonl).

Reports:
  - Overall Accuracy
  - Auto-Handle Precision, Recall, F1-Score
  - Escalation Precision, Recall, F1-Score
  - Confusion Matrix (2x2)
  - Breakdown by Difficulty Tier (easy, medium, hard)
  - Breakdown by Intent Category (all 10 intents)
  - Real-world case studies for correct decisions and edge cases
  - Tradeoff analysis between automation rate and safety

Exports machine-readable metrics to:
  data/analysis/escalation_evaluation_metrics.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Any, List
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.retrieve_resolutions import ResolutionRetriever
from src.escalation_policy import (
    extract_decision_signals,
    make_routing_decision,
    DEFAULT_THRESHOLDS
)

GOLDEN_SET_PATH = "data/processed/golden_evaluation_set.jsonl"
CALIBRATION_PATH = "data/analysis/escalation_threshold_calibration.json"
OUTPUT_METRICS_PATH = "data/analysis/escalation_evaluation_metrics.json"

def run_golden_evaluation(
    golden_path: str = GOLDEN_SET_PATH,
    calibration_path: str = CALIBRATION_PATH,
    output_path: str = OUTPUT_METRICS_PATH
) -> Dict[str, Any]:
    print("=" * 80)
    print("MILESTONE 10: ESCALATION POLICY EVALUATION ON GOLDEN SET (N = 200)")
    print("=" * 80)

    # 1. Load Calibrated Thresholds
    if os.path.exists(calibration_path):
        with open(calibration_path, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
        thresholds = calib_data.get("selected_trustworthy_thresholds", DEFAULT_THRESHOLDS)
        print(f"Loaded validation-calibrated thresholds from: {calibration_path}")
    else:
        print(f"Warning: Calibration file '{calibration_path}' not found. Using default thresholds.")
        thresholds = DEFAULT_THRESHOLDS

    print("\nOperating Decision Thresholds:")
    for k, v in thresholds.items():
        print(f"  {k:<32} : {v}")

    # 2. Load Golden Evaluation Set
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden evaluation set missing at: {golden_path}")

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_records = [json.loads(line) for line in f if line.strip()]

    print(f"\nLoaded {len(golden_records)} golden evaluation examples.")

    # 3. Evaluate Every Sample
    retriever = ResolutionRetriever()
    evaluated_cases = []

    y_true = []
    y_pred = []

    print("\nRunning Multi-Signal Routing Policy on Golden Evaluation Set...")
    for idx, rec in enumerate(tqdm(golden_records, desc="Evaluating Escalation")):
        cid = rec["conversation_id"]
        msg = rec["customer_message"]
        gt_esc = rec["escalation_recommendation"]  # 'auto_handle' or 'escalate_human_agent'
        gt_label = "AUTO_HANDLE" if gt_esc == "auto_handle" else "ESCALATE"
        gt_intent = rec["intent"]
        diff = rec["difficulty"]

        # Run routing decision
        routing_res = make_routing_decision(
            text=msg,
            thresholds=thresholds,
            retriever=retriever
        )

        pred_decision = routing_res["decision"]
        reason = routing_res["reason"]
        signals = routing_res["signals"]

        y_true.append(gt_label)
        y_pred.append(pred_decision)

        evaluated_cases.append({
            "case_index": idx + 1,
            "conversation_id": cid,
            "customer_message": msg,
            "ground_truth_intent": gt_intent,
            "difficulty": diff,
            "ground_truth_decision": gt_label,
            "predicted_decision": pred_decision,
            "is_correct": bool(pred_decision == gt_label),
            "reason": reason,
            "signals": signals
        })

    # 4. Global Performance Metrics
    labels = ["AUTO_HANDLE", "ESCALATE"]
    acc = float(accuracy_score(y_true, y_pred))

    auto_p = float(precision_score(y_true, y_pred, pos_label="AUTO_HANDLE", zero_division=0))
    auto_r = float(recall_score(y_true, y_pred, pos_label="AUTO_HANDLE", zero_division=0))
    auto_f1 = float(f1_score(y_true, y_pred, pos_label="AUTO_HANDLE", zero_division=0))

    esc_p = float(precision_score(y_true, y_pred, pos_label="ESCALATE", zero_division=0))
    esc_r = float(recall_score(y_true, y_pred, pos_label="ESCALATE", zero_division=0))
    esc_f1 = float(f1_score(y_true, y_pred, pos_label="ESCALATE", zero_division=0))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp_auto = int(cm[0][0])
    fp_auto = int(cm[1][0])  # True was ESCALATE, predicted AUTO_HANDLE (Critical Safety Risk!)
    fn_auto = int(cm[0][1])  # True was AUTO_HANDLE, predicted ESCALATE (Safe conservative margin)
    tp_esc = int(cm[1][1])

    automation_rate = round((tp_auto + fp_auto) / len(y_true), 4)

    print("\n" + "=" * 80)
    print("GLOBAL ESCALATION PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"  Total Evaluation Examples:   {len(y_true)}")
    print(f"  Overall Accuracy:            {acc * 100:.2f}%")
    print(f"  Automation Rate:             {automation_rate * 100:.2f}% ({tp_auto + fp_auto}/{len(y_true)} cases)")
    print(f"  Macro F1-Score:              {macro_f1 * 100:.2f}%")
    print(f"  Weighted F1-Score:           {weighted_f1 * 100:.2f}%")
    print("-" * 80)
    print("  AUTO_HANDLE Class Performance:")
    print(f"    Precision:                 {auto_p * 100:.2f}% (Trustworthiness: % of bot responses that are truly safe)")
    print(f"    Recall:                    {auto_r * 100:.2f}% (% of safe cases successfully automated)")
    print(f"    F1-Score:                  {auto_f1 * 100:.2f}%")
    print(f"    Support:                   {y_true.count('AUTO_HANDLE')}")
    print("-" * 80)
    print("  ESCALATE Class Performance:")
    print(f"    Precision:                 {esc_p * 100:.2f}%")
    print(f"    Recall:                    {esc_r * 100:.2f}% (Safety Guardrail: % of risky cases caught)")
    print(f"    F1-Score:                  {esc_f1 * 100:.2f}%")
    print(f"    Support:                   {y_true.count('ESCALATE')}")
    print("=" * 80)

    print("\nCONFUSION MATRIX:")
    print("                      Predicted AUTO_HANDLE    Predicted ESCALATE")
    print(f"  Actual AUTO_HANDLE  {tp_auto:>20}    {fn_auto:>18} (Safe False Escalations)")
    print(f"  Actual ESCALATE     {fp_auto:>20}    {tp_esc:>18} (True Escalations)")
    print(f"  Critical False Auto-Handles: {fp_auto} ({fp_auto / (tp_auto + fp_auto) * 100:.2f}% of automated responses)")

    # 5. Breakdown by Difficulty Tier
    difficulty_breakdown = {}
    for diff in ["easy", "medium", "hard"]:
        sub = [c for c in evaluated_cases if c["difficulty"] == diff]
        if sub:
            sub_y_true = [c["ground_truth_decision"] for c in sub]
            sub_y_pred = [c["predicted_decision"] for c in sub]
            difficulty_breakdown[diff] = {
                "sample_count": len(sub),
                "accuracy": round(float(accuracy_score(sub_y_true, sub_y_pred)), 4),
                "auto_precision": round(float(precision_score(sub_y_true, sub_y_pred, pos_label="AUTO_HANDLE", zero_division=0)), 4),
                "auto_recall": round(float(recall_score(sub_y_true, sub_y_pred, pos_label="AUTO_HANDLE", zero_division=0)), 4),
                "esc_precision": round(float(precision_score(sub_y_true, sub_y_pred, pos_label="ESCALATE", zero_division=0)), 4),
                "esc_recall": round(float(recall_score(sub_y_true, sub_y_pred, pos_label="ESCALATE", zero_division=0)), 4),
                "automation_rate": round(sub_y_pred.count("AUTO_HANDLE") / len(sub), 4)
            }

    print("\n" + "=" * 80)
    print("BREAKDOWN BY DIFFICULTY TIER:")
    print("=" * 80)
    print(f"{'Tier':<10} | {'Count':<6} | {'Accuracy':<10} | {'Auto Prec':<10} | {'Auto Recall':<12} | {'Esc Recall':<12} | {'Auto Rate':<10}")
    print("-" * 80)
    for diff, m in difficulty_breakdown.items():
        print(f"{diff:<10} | {m['sample_count']:<6} | {m['accuracy']*100:>8.2f}% | {m['auto_precision']*100:>8.2f}% | {m['auto_recall']*100:>10.2f}% | {m['esc_recall']*100:>10.2f}% | {m['automation_rate']*100:>8.2f}%")

    # 6. Breakdown by Intent Category
    intent_breakdown = {}
    intents = sorted(list(set(c["ground_truth_intent"] for c in evaluated_cases)))
    for intent in intents:
        sub = [c for c in evaluated_cases if c["ground_truth_intent"] == intent]
        sub_y_true = [c["ground_truth_decision"] for c in sub]
        sub_y_pred = [c["predicted_decision"] for c in sub]
        intent_breakdown[intent] = {
            "sample_count": len(sub),
            "actual_auto_count": sub_y_true.count("AUTO_HANDLE"),
            "actual_esc_count": sub_y_true.count("ESCALATE"),
            "predicted_auto_count": sub_y_pred.count("AUTO_HANDLE"),
            "predicted_esc_count": sub_y_pred.count("ESCALATE"),
            "accuracy": round(float(accuracy_score(sub_y_true, sub_y_pred)), 4),
            "automation_rate": round(sub_y_pred.count("AUTO_HANDLE") / len(sub), 4)
        }

    # 7. Qualitative Case Studies
    true_auto_samples = [c for c in evaluated_cases if c["predicted_decision"] == "AUTO_HANDLE" and c["ground_truth_decision"] == "AUTO_HANDLE"][:3]
    true_esc_samples = [c for c in evaluated_cases if c["predicted_decision"] == "ESCALATE" and c["ground_truth_decision"] == "ESCALATE"][:3]
    false_esc_samples = [c for c in evaluated_cases if c["predicted_decision"] == "ESCALATE" and c["ground_truth_decision"] == "AUTO_HANDLE"][:3]
    false_auto_samples = [c for c in evaluated_cases if c["predicted_decision"] == "AUTO_HANDLE" and c["ground_truth_decision"] == "ESCALATE"][:3]

    output_data = {
        "milestone": "Milestone 10: Auto-Handle vs Human Escalation",
        "benchmark_set": "Golden Evaluation Set (data/processed/golden_evaluation_set.jsonl)",
        "total_samples": len(golden_records),
        "operating_thresholds": thresholds,
        "global_metrics": {
            "accuracy": acc,
            "automation_rate": automation_rate,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "auto_handle": {
                "precision": auto_p,
                "recall": auto_r,
                "f1": auto_f1,
                "support": y_true.count("AUTO_HANDLE")
            },
            "escalate": {
                "precision": esc_p,
                "recall": esc_r,
                "f1": esc_f1,
                "support": y_true.count("ESCALATE")
            },
            "confusion_matrix": {
                "true_auto_handle": tp_auto,
                "false_auto_handle": fp_auto,
                "true_escalate": tp_esc,
                "false_escalate": fn_auto
            }
        },
        "difficulty_tier_breakdown": difficulty_breakdown,
        "intent_breakdown": intent_breakdown,
        "qualitative_case_studies": {
            "true_auto_handle_examples": true_auto_samples,
            "true_escalate_examples": true_esc_samples,
            "false_escalate_examples": false_esc_samples,
            "false_auto_handle_examples": false_auto_samples
        },
        "all_predictions": evaluated_cases
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[SUCCESS] Evaluation metrics saved to: {output_path}")
    return output_data

def main():
    run_golden_evaluation()

if __name__ == "__main__":
    main()
