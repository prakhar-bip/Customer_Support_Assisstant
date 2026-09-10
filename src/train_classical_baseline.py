"""
Train and Evaluate Classical ML Baseline: TF-IDF + Logistic Regression.

This script:
  1. Loads clean conversations and excludes all 200 golden evaluation samples (zero data leakage).
  2. Extracts labeled training examples across the 10 customer support intents.
  3. Creates an 80/20 stratified train/validation split.
  4. Fits a TF-IDF Vectorizer + Multinomial Logistic Regression pipeline with balanced class weights.
  5. Serializes the trained pipeline artifact to models/intent_tfidf_logistic_regression.joblib.
  6. Evaluates blindly on data/processed/golden_evaluation_set.jsonl.
  7. Computes global metrics, per-intent breakdowns, confusion matrix, and deltas vs Majority Baseline.
  8. Exports results to data/analysis/baseline_logistic_regression_metrics.json.
"""

import json
import os
import re
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

CLEAN_DATA_PATH = "data/processed/amazonhelp_conversations_clean.csv"
GOLDEN_SET_PATH = "data/processed/golden_evaluation_set.jsonl"
MAJORITY_METRICS_PATH = "data/analysis/baseline_majority_metrics.json"
MODEL_OUTPUT_DIR = "models"
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, "intent_tfidf_logistic_regression.joblib")
OUTPUT_METRICS_PATH = "data/analysis/baseline_logistic_regression_metrics.json"

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

# Precedence hierarchy for training data label assignment
PRECEDENCE_ORDER = [
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

def train_and_evaluate():
    print("=" * 80)
    print("MILESTONE 7: CLASSICAL ML BASELINE (TF-IDF + LOGISTIC REGRESSION)")
    print("=" * 80)

    # 1. Load Golden Evaluation Set to enforce strict quarantine
    if not os.path.exists(GOLDEN_SET_PATH):
        print(f"Error: Golden evaluation set missing at {GOLDEN_SET_PATH}")
        sys.exit(1)

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_records = [json.loads(line) for line in f if line.strip()]

    golden_ids = set(r["conversation_id"] for r in golden_records)
    print(f"Loaded {len(golden_records)} golden evaluation examples (strictly isolated).")

    # 2. Load Clean Dataset and Isolate Training Pool
    if not os.path.exists(CLEAN_DATA_PATH):
        print(f"Error: Clean dataset missing at {CLEAN_DATA_PATH}")
        sys.exit(1)

    print(f"\nLoading clean dataset from {CLEAN_DATA_PATH}...")
    df_clean = pd.read_csv(CLEAN_DATA_PATH)
    total_clean = len(df_clean)

    df_train_pool = df_clean[~df_clean["conversation_id"].isin(golden_ids)].copy()
    overlap = set(df_train_pool["conversation_id"]).intersection(golden_ids)
    assert len(overlap) == 0, f"FATAL: Data leakage detected! {len(overlap)} IDs overlap."

    print(f"Total clean rows: {total_clean:,}")
    print(f"Training pool rows: {len(df_train_pool):,} (Zero leakage confirmed).")

    # 3. Label Training Data Using Verified Taxonomy Patterns
    from src.discover_intents import TAXONOMY_PATTERNS
    compiled = {k: re.compile(v, re.I) for k, v in TAXONOMY_PATTERNS.items()}

    print("\nAssigning ground-truth intent labels across training pool...")
    assigned_labels = []
    for msg in df_train_pool["customer_message"].fillna("").astype(str):
        hits = [k for k, p in compiled.items() if p.search(msg)]
        if len(hits) == 1:
            assigned_labels.append(hits[0])
        elif len(hits) > 1:
            assigned_labels.append(next((p for p in PRECEDENCE_ORDER if p in hits), hits[0]))
        else:
            assigned_labels.append(None)

    df_train_pool["intent"] = assigned_labels
    labeled_train = df_train_pool.dropna(subset=["intent"]).copy()
    print(f"Total labeled training examples: {len(labeled_train):,}")

    print("\nTraining Class Distribution:")
    for intent, cnt in labeled_train["intent"].value_counts().items():
        print(f"  {intent:<40} : {cnt:>5} ({cnt/len(labeled_train)*100:>5.1f}%)")

    # 4. Stratified Train / Validation Split (80% Train, 20% Validation)
    X = labeled_train["customer_message"].astype(str)
    y = labeled_train["intent"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\nPartitioned data: Train={len(X_train):,} ({len(X_train)/len(X)*100:.0f}%), Validation={len(X_val):,} ({len(X_val)/len(X)*100:.0f}%)")

    # 5. Build Pipeline: TF-IDF + Balanced Logistic Regression
    print("\nFitting TF-IDF Vectorizer + Multinomial Logistic Regression Pipeline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs"
        ))
    ])

    pipeline.fit(X_train, y_train)
    print("[SUCCESS] Pipeline fitted successfully.")

    # 6. Evaluate on Validation Split
    val_preds = pipeline.predict(X_val)
    val_acc = float(accuracy_score(y_val, val_preds))
    val_f1_macro = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    print(f"\n[VALIDATION PERFORMANCE] Accuracy: {val_acc * 100:.2f}% | Macro F1: {val_f1_macro * 100:.2f}%")

    # 7. Serialize Trained Model Artifact
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"[SUCCESS] Serialized trained pipeline to {MODEL_OUTPUT_PATH}")

    # 8. Evaluate Blindly on Golden Evaluation Set
    print("\n" + "=" * 80)
    print("BLIND EVALUATION ON GOLDEN EVALUATION SET (N = 200)")
    print("=" * 80)

    X_gold = [r["customer_message"] for r in golden_records]
    y_gold = [r["intent"] for r in golden_records]

    gold_preds = pipeline.predict(X_gold)

    # Compute Global Metrics
    gold_acc = float(accuracy_score(y_gold, gold_preds))
    gold_macro_prec = float(precision_score(y_gold, gold_preds, labels=TAXONOMY_INTENTS, average="macro", zero_division=0))
    gold_weighted_prec = float(precision_score(y_gold, gold_preds, labels=TAXONOMY_INTENTS, average="weighted", zero_division=0))
    gold_macro_rec = float(recall_score(y_gold, gold_preds, labels=TAXONOMY_INTENTS, average="macro", zero_division=0))
    gold_weighted_rec = float(recall_score(y_gold, gold_preds, labels=TAXONOMY_INTENTS, average="weighted", zero_division=0))
    gold_macro_f1 = float(f1_score(y_gold, gold_preds, labels=TAXONOMY_INTENTS, average="macro", zero_division=0))
    gold_weighted_f1 = float(f1_score(y_gold, gold_preds, labels=TAXONOMY_INTENTS, average="weighted", zero_division=0))

    # Per-intent metrics
    gold_report = classification_report(
        y_gold,
        gold_preds,
        labels=TAXONOMY_INTENTS,
        target_names=TAXONOMY_INTENTS,
        output_dict=True,
        zero_division=0
    )

    per_intent_metrics = {}
    for intent in TAXONOMY_INTENTS:
        m = gold_report[intent]
        per_intent_metrics[intent] = {
            "precision": round(float(m["precision"]), 4),
            "recall": round(float(m["recall"]), 4),
            "f1_score": round(float(m["f1-score"]), 4),
            "support": int(m["support"])
        }

    # Confusion Matrix
    cm = confusion_matrix(y_gold, gold_preds, labels=TAXONOMY_INTENTS)
    cm_dict = {
        "labels": TAXONOMY_INTENTS,
        "matrix": cm.tolist()
    }

    # Print Global Results
    print(f"Accuracy:           {gold_acc * 100:.2f}%  (Correct: {int(gold_acc * len(y_gold))} / {len(y_gold)})")
    print(f"Macro Precision:    {gold_macro_prec * 100:.2f}%")
    print(f"Weighted Precision: {gold_weighted_prec * 100:.2f}%")
    print(f"Macro Recall:       {gold_macro_rec * 100:.2f}%")
    print(f"Weighted Recall:    {gold_weighted_rec * 100:.2f}%")
    print(f"Macro F1-Score:     {gold_macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score:  {gold_weighted_f1 * 100:.2f}%")

    # Print Per-Intent Table
    print("\n" + "=" * 80)
    print(f"{'Intent Name':<42} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'Support':>7}")
    print("-" * 80)
    for intent in TAXONOMY_INTENTS:
        m = per_intent_metrics[intent]
        print(
            f"{intent:<42} | "
            f"{m['precision'] * 100:>6.2f}% | "
            f"{m['recall'] * 100:>6.2f}% | "
            f"{m['f1_score'] * 100:>6.2f}% | "
            f"{m['support']:>7}"
        )
    print("=" * 80)

    # Print Confusion Matrix
    print("\nCONFUSION MATRIX (Rows: Actual Ground Truth, Columns: Predicted):")
    print(f"{'Actual Intent':<40} | " + " | ".join(f"{i+1:>2}" for i in range(len(TAXONOMY_INTENTS))))
    print("-" * 75)
    for i, intent in enumerate(TAXONOMY_INTENTS):
        row_str = " | ".join(f"{cm[i][j]:>2}" for j in range(len(TAXONOMY_INTENTS)))
        print(f"[{i+1:>2}] {intent:<36} | {row_str}")
    print("-" * 75)

    # 9. Comparative Benchmark vs. Majority Baseline (Milestone 6)
    delta_metrics = {}
    if os.path.exists(MAJORITY_METRICS_PATH):
        with open(MAJORITY_METRICS_PATH, "r", encoding="utf-8") as f:
            majority_data = json.load(f)
        maj_m = majority_data.get("metrics", {})
        delta_metrics = {
            "accuracy_delta": round(gold_acc - maj_m.get("accuracy", 0), 4),
            "macro_precision_delta": round(gold_macro_prec - maj_m.get("macro_precision", 0), 4),
            "macro_recall_delta": round(gold_macro_rec - maj_m.get("macro_recall", 0), 4),
            "macro_f1_delta": round(gold_macro_f1 - maj_m.get("macro_f1", 0), 4),
            "weighted_f1_delta": round(gold_weighted_f1 - maj_m.get("weighted_f1", 0), 4)
        }
        print("\n" + "=" * 80)
        print("COMPARISON: CLASSICAL ML BASELINE vs. MAJORITY BASELINE")
        print("=" * 80)
        print(f"{'Metric':<25} | {'Majority Baseline':>18} | {'Classical ML':>14} | {'Absolute Gain':>14}")
        print("-" * 80)
        print(f"{'Accuracy':<25} | {maj_m.get('accuracy', 0)*100:>17.2f}% | {gold_acc*100:>13.2f}% | {delta_metrics['accuracy_delta']*100:>+13.2f}%")
        print(f"{'Macro Precision':<25} | {maj_m.get('macro_precision', 0)*100:>17.2f}% | {gold_macro_prec*100:>13.2f}% | {delta_metrics['macro_precision_delta']*100:>+13.2f}%")
        print(f"{'Macro Recall':<25} | {maj_m.get('macro_recall', 0)*100:>17.2f}% | {gold_macro_rec*100:>13.2f}% | {delta_metrics['macro_recall_delta']*100:>+13.2f}%")
        print(f"{'Macro F1-Score':<25} | {maj_m.get('macro_f1', 0)*100:>17.2f}% | {gold_macro_f1*100:>13.2f}% | {delta_metrics['macro_f1_delta']*100:>+13.2f}%")
        print(f"{'Weighted F1-Score':<25} | {maj_m.get('weighted_f1', 0)*100:>17.2f}% | {gold_weighted_f1*100:>13.2f}% | {delta_metrics['weighted_f1_delta']*100:>+13.2f}%")
        print("=" * 80)

    # 10. Save Machine-Readable Metrics
    output_payload = {
        "benchmark_name": "Milestone 7 Classical ML Baseline",
        "model_architecture": {
            "pipeline": "TfidfVectorizer + LogisticRegression",
            "vectorizer": {
                "ngram_range": [1, 2],
                "max_features": 10000,
                "sublinear_tf": True,
                "stop_words": "english"
            },
            "classifier": {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "random_state": 42
            }
        },
        "dataset_splits": {
            "clean_total": total_clean,
            "golden_quarantined": len(golden_ids),
            "training_pool": len(labeled_train),
            "train_split": len(X_train),
            "validation_split": len(X_val)
        },
        "validation_performance": {
            "accuracy": round(val_acc, 4),
            "macro_f1": round(val_f1_macro, 4)
        },
        "golden_set_performance": {
            "accuracy": round(gold_acc, 4),
            "macro_precision": round(gold_macro_prec, 4),
            "weighted_precision": round(gold_weighted_prec, 4),
            "macro_recall": round(gold_macro_rec, 4),
            "weighted_recall": round(gold_weighted_rec, 4),
            "macro_f1": round(gold_macro_f1, 4),
            "weighted_f1": round(gold_weighted_f1, 4)
        },
        "per_intent_metrics": per_intent_metrics,
        "confusion_matrix": cm_dict,
        "comparison_vs_majority_baseline": delta_metrics,
        "model_artifact_path": MODEL_OUTPUT_PATH
    }

    os.makedirs(os.path.dirname(OUTPUT_METRICS_PATH), exist_ok=True)
    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[SUCCESS] Exported machine-readable metrics to {OUTPUT_METRICS_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
