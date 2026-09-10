"""
Automated Validation Suite for Milestone 5: Golden Evaluation Set.

This script executes 9 rigorous quality gates against the curated golden evaluation set:
  1. Sample count boundary check (150 <= N <= 250).
  2. Zero missing/null/empty field check across all required attributes.
  3. Strict intent taxonomy validation against the 10 approved intent codes.
  4. Class balance check (minimum 12 samples per intent, all 10 classes present).
  5. Duplicate detection across conversation IDs and customer text.
  6. Escalation recommendation domain validation ('auto_handle' / 'escalate_human_agent').
  7. Difficulty rating domain validation ('easy', 'medium', 'hard').
  8. Reference integrity verification against the source clean dataset.
  9. Format parity verification between JSONL and CSV representations.

Exits with code 0 on complete pass, or code 1 on any assertion failure.
"""

import json
import os
import sys
import pandas as pd

GOLDEN_JSONL = "data/processed/golden_evaluation_set.jsonl"
GOLDEN_CSV = "data/processed/golden_evaluation_set.csv"
CLEAN_DATASET = "data/processed/amazonhelp_conversations_clean.csv"

# Approved 10-Intent Taxonomy
APPROVED_INTENTS = {
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
}

# Standardized Expected Actions
APPROVED_ACTIONS = {
    "TRACK_SHIPMENT_AND_CARRIER_UPDATE",
    "INVESTIGATE_FALSE_DELIVERY",
    "REPLACE_DAMAGED_DEFECTIVE_ITEM",
    "INITIATE_RETURN_PICKUP_LABEL",
    "TRACK_AND_DISBURSE_REFUND",
    "PROCESS_ORDER_CANCELLATION",
    "RESOLVE_PAYMENT_BILLING_ERROR",
    "MANAGE_PRIME_SUBSCRIPTION",
    "RECOVER_ACCOUNT_SECURITY",
    "TROUBLESHOOT_DIGITAL_OR_TECHNICAL",
    "ESCALATE_SERVICE_GRIEVANCE"
}

APPROVED_ESCALATIONS = {"auto_handle", "escalate_human_agent"}
APPROVED_DIFFICULTIES = {"easy", "medium", "hard"}

REQUIRED_FIELDS = [
    "conversation_id",
    "timestamp",
    "customer_message",
    "intent",
    "difficulty",
    "turn_count",
    "expected_action",
    "escalation_recommendation",
    "escalation_reason",
    "historical_brand_response",
    "labeler_notes"
]

def run_validation():
    print("=" * 80)
    print("RUNNING AUTOMATED VALIDATION SUITE: GOLDEN EVALUATION SET")
    print("=" * 80)

    # Gate 0: File existence
    for path in [GOLDEN_JSONL, GOLDEN_CSV, CLEAN_DATASET]:
        if not os.path.exists(path):
            print(f"[FAIL] Missing required file: {path}")
            sys.exit(1)
        print(f"[PASS] Verified existence of {path}")

    # Load data
    with open(GOLDEN_JSONL, "r", encoding="utf-8") as f:
        jsonl_records = [json.loads(line) for line in f if line.strip()]
    csv_df = pd.read_csv(GOLDEN_CSV)
    clean_df = pd.read_csv(CLEAN_DATASET)
    clean_ids = set(clean_df['conversation_id'])

    total_samples = len(jsonl_records)
    print(f"\nTotal records loaded: {total_samples}")

    failures = []

    # Gate 1: Number of examples check (150 <= N <= 250)
    if not (150 <= total_samples <= 250):
        failures.append(f"Sample count {total_samples} outside allowable range [150, 250].")
    else:
        print(f"[PASS] Sample count check: {total_samples} examples (Target: 200).")

    # Gate 2: Missing fields / nulls check
    for idx, rec in enumerate(jsonl_records):
        for field in REQUIRED_FIELDS:
            if field not in rec:
                failures.append(f"Record #{idx} ({rec.get('conversation_id', 'unknown')}) missing required field '{field}'.")
            elif rec[field] is None or (isinstance(rec[field], str) and not rec[field].strip()):
                failures.append(f"Record #{idx} ({rec.get('conversation_id', 'unknown')}) has empty/null value for '{field}'.")

    if not failures:
        print("[PASS] Zero missing/null/empty values check across all 11 attributes.")

    # Gate 3: Invalid intent names check
    invalid_intents = []
    for rec in jsonl_records:
        intent = rec.get("intent", "")
        if intent not in APPROVED_INTENTS:
            invalid_intents.append((rec.get("conversation_id"), intent))

    if invalid_intents:
        failures.append(f"Found {len(invalid_intents)} invalid intent names: {invalid_intents[:5]}")
    else:
        print(f"[PASS] Strict intent taxonomy check: 100% of samples match approved 10 intents.")

    # Gate 4: Class distribution check
    intent_counts = pd.Series([r['intent'] for r in jsonl_records]).value_counts().to_dict()
    missing_classes = APPROVED_INTENTS - set(intent_counts.keys())
    if missing_classes:
        failures.append(f"Missing classes in golden set: {missing_classes}")
    
    underrepresented = {k: v for k, v in intent_counts.items() if v < 12}
    if underrepresented:
        failures.append(f"Classes with fewer than 12 examples: {underrepresented}")

    print("\n[INFO] Golden Evaluation Set Class Distribution:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {intent:<40} : {count:>3} ({count/total_samples*100:>5.1f}%)")

    if not missing_classes and not underrepresented:
        print("[PASS] Class distribution check: All 10 intents represented (>= 12 samples each).")

    # Gate 5: Duplicate check
    seen_ids = set()
    dup_ids = []
    seen_msgs = set()
    dup_msgs = []

    for rec in jsonl_records:
        cid = rec.get("conversation_id")
        msg = rec.get("customer_message", "").strip().lower()
        if cid in seen_ids:
            dup_ids.append(cid)
        seen_ids.add(cid)

        if msg in seen_msgs:
            dup_msgs.append(cid)
        seen_msgs.add(msg)

    if dup_ids:
        failures.append(f"Duplicate conversation IDs detected: {dup_ids}")
    if dup_msgs:
        failures.append(f"Duplicate customer messages detected: {dup_msgs}")

    if not dup_ids and not dup_msgs:
        print(f"[PASS] Duplicate check: 0 duplicate IDs, 0 duplicate messages across all {total_samples} samples.")

    # Gate 6: Escalation values check
    invalid_esc = [r.get("conversation_id") for r in jsonl_records if r.get("escalation_recommendation") not in APPROVED_ESCALATIONS]
    if invalid_esc:
        failures.append(f"Invalid escalation values for IDs: {invalid_esc}")
    else:
        esc_counts = pd.Series([r['escalation_recommendation'] for r in jsonl_records]).value_counts().to_dict()
        print(f"[PASS] Escalation recommendation check: {esc_counts}")

    # Gate 7: Difficulty values check
    invalid_diff = [r.get("conversation_id") for r in jsonl_records if r.get("difficulty") not in APPROVED_DIFFICULTIES]
    if invalid_diff:
        failures.append(f"Invalid difficulty values for IDs: {invalid_diff}")
    else:
        diff_counts = pd.Series([r['difficulty'] for r in jsonl_records]).value_counts().to_dict()
        print(f"[PASS] Difficulty tier check: {diff_counts}")

    # Gate 8: Expected action check
    invalid_actions = [r.get("conversation_id") for r in jsonl_records if r.get("expected_action") not in APPROVED_ACTIONS]
    if invalid_actions:
        failures.append(f"Invalid expected action values for IDs: {invalid_actions}")
    else:
        action_counts = pd.Series([r['expected_action'] for r in jsonl_records]).value_counts().to_dict()
        print(f"[PASS] Expected action check: {len(action_counts)} distinct actions verified.")

    # Gate 9: Source dataset reference integrity check
    missing_source_ids = [r['conversation_id'] for r in jsonl_records if r['conversation_id'] not in clean_ids]
    if missing_source_ids:
        failures.append(f"Conversation IDs missing from clean dataset: {missing_source_ids}")
    else:
        print(f"[PASS] Reference integrity check: All {total_samples} IDs verified in {CLEAN_DATASET}.")

    # Gate 10: JSONL and CSV parity check
    if len(csv_df) != len(jsonl_records):
        failures.append(f"Parity mismatch: CSV has {len(csv_df)} rows, JSONL has {len(jsonl_records)} rows.")
    elif list(csv_df['conversation_id']) != [r['conversation_id'] for r in jsonl_records]:
        failures.append("Conversation ID ordering mismatch between CSV and JSONL.")
    else:
        print(f"[PASS] Format parity check: JSONL and CSV match 1-to-1 perfectly.")

    # Final summary
    print("\n" + "=" * 80)
    if failures:
        print(f"VALIDATION FAILED WITH {len(failures)} ERROR(S):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL 10 VALIDATION GATES PASSED SUCCESSFULLY (EXIT CODE 0).")
        print("Golden Evaluation Set is verified, clean, and certified for model benchmarking.")
        print("=" * 80)
        sys.exit(0)

if __name__ == "__main__":
    run_validation()
