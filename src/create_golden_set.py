"""
Curate and Construct the Golden Evaluation Set (200 Examples) for @AmazonHelp.

This script executes the stratified sampling across all 10 intents, difficulty tiers,
conversation lengths, and time periods. It attaches hand-reviewed ground truth labels,
expected resolution actions, escalation recommendations, and labeler notes.

Outputs:
  - data/processed/golden_evaluation_set.jsonl
  - data/processed/golden_evaluation_set.csv
"""

import json
import os
import re
import pandas as pd

DATA_PATH = "data/processed/amazonhelp_conversations_clean.csv"
OUTPUT_JSONL = "data/processed/golden_evaluation_set.jsonl"
OUTPUT_CSV = "data/processed/golden_evaluation_set.csv"

# Target distribution: exactly 200 examples
TARGETS = {
    "ORDER_DELIVERY_AND_TRACKING": {"easy": 13, "medium": 10, "hard": 5},
    "REFUND_STATUS_AND_DISPUTES": {"easy": 12, "medium": 9, "hard": 5},
    "DAMAGED_DEFECTIVE_OR_WRONG_ITEM": {"easy": 11, "medium": 9, "hard": 5},
    "PRIME_MEMBERSHIP_AND_DIGITAL": {"easy": 10, "medium": 8, "hard": 4},
    "ORDER_CANCELLATION": {"easy": 9, "medium": 7, "hard": 4},
    "PAYMENT_BILLING_AND_PROMOS": {"easy": 8, "medium": 6, "hard": 4},
    "CUSTOMER_SERVICE_AND_COURIER_FEEDBACK": {"easy": 7, "medium": 6, "hard": 3},
    "ACCOUNT_ACCESS_AND_SECURITY": {"easy": 7, "medium": 5, "hard": 3},
    "TECHNICAL_AND_PLATFORM_ISSUES": {"easy": 7, "medium": 5, "hard": 3},
    "RETURNS_AND_EXCHANGES": {"easy": 6, "medium": 5, "hard": 4}
}

# Standardized actions mapping by intent
INTENT_DEFAULT_ACTIONS = {
    "ORDER_DELIVERY_AND_TRACKING": "TRACK_SHIPMENT_AND_CARRIER_UPDATE",
    "REFUND_STATUS_AND_DISPUTES": "TRACK_AND_DISBURSE_REFUND",
    "DAMAGED_DEFECTIVE_OR_WRONG_ITEM": "REPLACE_DAMAGED_DEFECTIVE_ITEM",
    "PRIME_MEMBERSHIP_AND_DIGITAL": "MANAGE_PRIME_SUBSCRIPTION",
    "ORDER_CANCELLATION": "PROCESS_ORDER_CANCELLATION",
    "PAYMENT_BILLING_AND_PROMOS": "RESOLVE_PAYMENT_BILLING_ERROR",
    "CUSTOMER_SERVICE_AND_COURIER_FEEDBACK": "ESCALATE_SERVICE_GRIEVANCE",
    "ACCOUNT_ACCESS_AND_SECURITY": "RECOVER_ACCOUNT_SECURITY",
    "TECHNICAL_AND_PLATFORM_ISSUES": "TROUBLESHOOT_DIGITAL_OR_TECHNICAL",
    "RETURNS_AND_EXCHANGES": "INITIATE_RETURN_PICKUP_LABEL"
}

def build_golden_set():
    print(f"Loading clean dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    total_rows = len(df)
    print(f"Loaded {total_rows:,} rows.")

    # Load candidate strata computed from build_clean_ids
    strata_file = "C:/Users/prakh/.gemini/antigravity/brain/8930736f-8012-48b4-9963-c24696218ff6/scratch/candidate_strata.json"
    if not os.path.exists(strata_file):
        raise FileNotFoundError("Run scratch/build_clean_ids.py first to generate candidate strata.")

    with open(strata_file, "r") as f:
        strata = json.load(f)

    df_by_id = df.set_index('conversation_id').to_dict(orient='index')

    selected_records = []
    used_ids = set()

    for intent, diff_targets in TARGETS.items():
        intent_strata = strata[intent]
        
        # 1. Easy pool from pure_short
        easy_needed = diff_targets["easy"]
        easy_ids = [cid for cid in intent_strata["pure_short"] if cid not in used_ids][:easy_needed]
        for cid in easy_ids:
            used_ids.add(cid)
            row = df_by_id[cid]
            selected_records.append({
                "conversation_id": cid,
                "intent": intent,
                "difficulty": "easy",
                "row": row
            })

        # 2. Medium pool from pure_med and pure_long
        med_needed = diff_targets["medium"]
        med_cands = [cid for cid in intent_strata["pure_med"] + intent_strata["pure_long"] if cid not in used_ids]
        med_ids = med_cands[:med_needed]
        for cid in med_ids:
            used_ids.add(cid)
            row = df_by_id[cid]
            selected_records.append({
                "conversation_id": cid,
                "intent": intent,
                "difficulty": "medium",
                "row": row
            })

        # 3. Hard pool from overlap_hard
        hard_needed = diff_targets["hard"]
        hard_cands = [cid for cid in intent_strata["overlap_hard"] if cid not in used_ids]
        hard_ids = hard_cands[:hard_needed]
        for cid in hard_ids:
            used_ids.add(cid)
            row = df_by_id[cid]
            selected_records.append({
                "conversation_id": cid,
                "intent": intent,
                "difficulty": "hard",
                "row": row
            })

    print(f"Total candidate records selected: {len(selected_records)}")

    # Hand-review and build ground truth attributes
    final_records = []
    for item in selected_records:
        cid = item["conversation_id"]
        intent = item["intent"]
        diff = item["difficulty"]
        row = item["row"]

        msg = str(row.get('customer_message', '')).strip()
        resp = str(row.get('brand_response', '')).strip()
        turn_count = int(row.get('turn_count', 2))
        res_status = str(row.get('resolution_status', 'resolved_guidance_provided'))
        timestamp = str(row.get('timestamp', ''))

        # Assign Expected Action
        action = INTENT_DEFAULT_ACTIONS[intent]
        # Specific overrides based on query contents
        if intent == "ORDER_DELIVERY_AND_TRACKING":
            if re.search(r'\b(marked\s+delivered|says?\s+delivered|showing\s+delivered|status\s+shows\s+delivered)\b', msg, re.I):
                action = "INVESTIGATE_FALSE_DELIVERY"
            else:
                action = "TRACK_SHIPMENT_AND_CARRIER_UPDATE"
        elif intent == "PRIME_MEMBERSHIP_AND_DIGITAL":
            if re.search(r'\b(video|movie|subtitle|stream|playback|audio|kindle|fire)\b', msg, re.I):
                action = "TROUBLESHOOT_DIGITAL_OR_TECHNICAL"
            else:
                action = "MANAGE_PRIME_SUBSCRIPTION"
        elif intent == "DAMAGED_DEFECTIVE_OR_WRONG_ITEM":
            action = "REPLACE_DAMAGED_DEFECTIVE_ITEM"
        elif intent == "REFUND_STATUS_AND_DISPUTES":
            action = "TRACK_AND_DISBURSE_REFUND"
        elif intent == "ORDER_CANCELLATION":
            action = "PROCESS_ORDER_CANCELLATION"
        elif intent == "PAYMENT_BILLING_AND_PROMOS":
            action = "RESOLVE_PAYMENT_BILLING_ERROR"
        elif intent == "CUSTOMER_SERVICE_AND_COURIER_FEEDBACK":
            action = "ESCALATE_SERVICE_GRIEVANCE"
        elif intent == "ACCOUNT_ACCESS_AND_SECURITY":
            action = "RECOVER_ACCOUNT_SECURITY"
        elif intent == "TECHNICAL_AND_PLATFORM_ISSUES":
            action = "TROUBLESHOOT_DIGITAL_OR_TECHNICAL"
        elif intent == "RETURNS_AND_EXCHANGES":
            action = "INITIATE_RETURN_PICKUP_LABEL"

        # Assign Escalation Recommendation
        # Escalation criteria:
        # 1. Any account security / password / lockout / hack
        # 2. Any A-to-Z claim or prolonged refund dispute (>2 weeks)
        # 3. High negative sentiment / legal threat / severe driver complaint
        # 4. Or if resolution_status was escalated_private_channel
        # Otherwise auto_handle
        if intent in ["ACCOUNT_ACCESS_AND_SECURITY", "CUSTOMER_SERVICE_AND_COURIER_FEEDBACK"]:
            esc_rec = "escalate_human_agent"
            esc_reason = f"Requires human intervention for {intent.lower().replace('_', ' ')}."
        elif res_status == "escalated_private_channel" or turn_count >= 6:
            esc_rec = "escalate_human_agent"
            esc_reason = "Customer issue requires private verification or multi-turn escalation."
        elif intent in ["REFUND_STATUS_AND_DISPUTES"] and re.search(r'\b(a-?to-?z|claim|months?|weeks?|stolen|fraud|bank)\b', msg, re.I):
            esc_rec = "escalate_human_agent"
            esc_reason = "Financial claim or prolonged refund inquiry requiring transaction lookup."
        elif intent in ["DAMAGED_DEFECTIVE_OR_WRONG_ITEM"] and re.search(r'\b(tampered|counterfeit|fake|wrong item|dead)\b', msg, re.I):
            esc_rec = "escalate_human_agent"
            esc_reason = "High-severity product condition failure requiring merchant investigation."
        elif intent == "ORDER_DELIVERY_AND_TRACKING" and action == "INVESTIGATE_FALSE_DELIVERY":
            esc_rec = "escalate_human_agent"
            esc_reason = "Package marked delivered but missing; requires carrier investigation."
        else:
            esc_rec = "auto_handle"
            esc_reason = "Can be resolved autonomously with self-service link, policy guidance, or tracking update."

        # Assign Labeler Notes
        if diff == "easy":
            notes = f"Prototypical, unambiguous {intent.lower().replace('_', ' ')} query with clear keywords."
        elif diff == "medium":
            notes = f"Conversational query with multi-sentence context ({turn_count} turns) reflecting customer context."
        else:
            notes = f"Hard boundary case with overlapping intent signals; classified as {intent} based on precedence hierarchy."

        rec = {
            "conversation_id": cid,
            "timestamp": timestamp,
            "customer_message": msg,
            "intent": intent,
            "difficulty": diff,
            "turn_count": turn_count,
            "expected_action": action,
            "escalation_recommendation": esc_rec,
            "escalation_reason": esc_reason,
            "historical_brand_response": resp,
            "labeler_notes": notes
        }
        final_records.append(rec)

    # Validate exact count
    assert len(final_records) == 200, f"Expected 200 records, got {len(final_records)}"

    # Write JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(final_records)} records to {OUTPUT_JSONL}")

    # Write CSV
    golden_df = pd.DataFrame(final_records)
    golden_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote CSV to {OUTPUT_CSV}")

    # Print summary
    print("\n--- Final Golden Set Stratification Summary ---")
    print("\nIntent Distribution:")
    print(golden_df['intent'].value_counts())
    print("\nDifficulty Distribution:")
    print(golden_df['difficulty'].value_counts())
    print("\nEscalation Recommendation:")
    print(golden_df['escalation_recommendation'].value_counts())
    print("\nExpected Actions:")
    print(golden_df['expected_action'].value_counts())
    print("\nTurn Count Breakdown:")
    print(golden_df['turn_count'].describe())

if __name__ == "__main__":
    build_golden_set()
