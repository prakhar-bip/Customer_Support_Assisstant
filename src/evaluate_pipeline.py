"""
End-to-End Pipeline Evaluation & Verification Suite (@AmazonHelp).

Runs the complete hybrid customer support agent pipeline on 20 diverse
sample inquiries covering:
  - All 10 intent taxonomy categories
  - Multi-difficulty tiers (easy, medium, hard)
  - Safe auto-handle queries vs safety-critical escalations
  - Boundary cases and edge inputs

Verifies that every stage (preprocessing, classification, retrieval,
escalation routing, response generation) produces valid, schema-compliant output.

Exports results to:
  data/analysis/pipeline_verification_results.json
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.pipeline import HybridSupportAgent

SAMPLE_TEST_QUERIES = [
    # 1. ORDER_DELIVERY_AND_TRACKING (Easy / Auto-Handle)
    {
        "id": "sample_01",
        "category": "ORDER_DELIVERY_AND_TRACKING",
        "expected_decision": "AUTO_HANDLE",
        "message": "Where is my parcel? The tracking status says out for delivery for 6 hours."
    },
    # 2. ORDER_DELIVERY_AND_TRACKING (False Delivery Dispute / Escalate)
    {
        "id": "sample_02",
        "category": "ORDER_DELIVERY_AND_TRACKING",
        "expected_decision": "ESCALATE",
        "message": "Order says delivered but I never received it! Checked everywhere, nowhere to be found."
    },
    # 3. REFUND_STATUS_AND_DISPUTES (Easy / Auto-Handle)
    {
        "id": "sample_03",
        "category": "REFUND_STATUS_AND_DISPUTES",
        "expected_decision": "AUTO_HANDLE",
        "message": "How long does it take for a refund to show on my credit card after cancellation?"
    },
    # 4. REFUND_STATUS_AND_DISPUTES (Prolonged dispute / Escalate)
    {
        "id": "sample_04",
        "category": "REFUND_STATUS_AND_DISPUTES",
        "expected_decision": "ESCALATE",
        "message": "It has been over 3 weeks and multiple chats, still no refund for order 402-1234567-8901234!"
    },
    # 5. DAMAGED_DEFECTIVE_OR_WRONG_ITEM (Easy / Auto-Handle)
    {
        "id": "sample_05",
        "category": "DAMAGED_DEFECTIVE_OR_WRONG_ITEM",
        "expected_decision": "AUTO_HANDLE",
        "message": "Received a broken ceramic mug in my delivery today. How can I get a replacement?"
    },
    # 6. DAMAGED_DEFECTIVE_OR_WRONG_ITEM (High Severity / Escalate)
    {
        "id": "sample_06",
        "category": "DAMAGED_DEFECTIVE_OR_WRONG_ITEM",
        "expected_decision": "ESCALATE",
        "message": "The package arrived with seal tampered and empty box! The phone inside was stolen."
    },
    # 7. PRIME_MEMBERSHIP_AND_DIGITAL (Easy / Auto-Handle)
    {
        "id": "sample_07",
        "category": "PRIME_MEMBERSHIP_AND_DIGITAL",
        "expected_decision": "AUTO_HANDLE",
        "message": "How do I cancel my Amazon Prime membership auto-renewal before next month?"
    },
    # 8. PRIME_MEMBERSHIP_AND_DIGITAL (Streaming / Auto-Handle)
    {
        "id": "sample_08",
        "category": "PRIME_MEMBERSHIP_AND_DIGITAL",
        "expected_decision": "AUTO_HANDLE",
        "message": "Subtitles are out of sync on Prime Video app when watching on Samsung TV."
    },
    # 9. ORDER_CANCELLATION (Easy / Auto-Handle)
    {
        "id": "sample_09",
        "category": "ORDER_CANCELLATION",
        "expected_decision": "AUTO_HANDLE",
        "message": "Ordered the wrong size by mistake 5 minutes ago. How do I cancel the order?"
    },
    # 10. ORDER_CANCELLATION (Late dispatch lockout / Escalate)
    {
        "id": "sample_10",
        "category": "ORDER_CANCELLATION",
        "expected_decision": "ESCALATE",
        "message": "Tried to cancel my order but the button is greyed out saying dispatching soon."
    },
    # 11. PAYMENT_BILLING_AND_PROMOS (Amazon Pay / Auto-Handle)
    {
        "id": "sample_11",
        "category": "PAYMENT_BILLING_AND_PROMOS",
        "expected_decision": "AUTO_HANDLE",
        "message": "Can I transfer my Amazon Pay balance back to my bank account?"
    },
    # 12. PAYMENT_BILLING_AND_PROMOS (Duplicate / Chargeback / Escalate)
    {
        "id": "sample_12",
        "category": "PAYMENT_BILLING_AND_PROMOS",
        "expected_decision": "ESCALATE",
        "message": "I was charged twice on my card for the same purchase! Will dispute with my bank."
    },
    # 13. CUSTOMER_SERVICE_AND_COURIER_FEEDBACK (Driver Abuse / Escalate)
    {
        "id": "sample_13",
        "category": "CUSTOMER_SERVICE_AND_COURIER_FEEDBACK",
        "expected_decision": "ESCALATE",
        "message": "Delivery driver was extremely rude and threw the box across my front porch!"
    },
    # 14. CUSTOMER_SERVICE_AND_COURIER_FEEDBACK (Agent Frustration / Escalate)
    {
        "id": "sample_14",
        "category": "CUSTOMER_SERVICE_AND_COURIER_FEEDBACK",
        "expected_decision": "ESCALATE",
        "message": "Your phone agent just hung up on me after keeping me on hold for 40 minutes!"
    },
    # 15. ACCOUNT_ACCESS_AND_SECURITY (Compromise / Fraud / Escalate)
    {
        "id": "sample_15",
        "category": "ACCOUNT_ACCESS_AND_SECURITY",
        "expected_decision": "ESCALATE",
        "message": "My seller account was hacked and unauthorized orders were placed! Please lock it now."
    },
    # 16. ACCOUNT_ACCESS_AND_SECURITY (OTP lockout / Escalate)
    {
        "id": "sample_16",
        "category": "ACCOUNT_ACCESS_AND_SECURITY",
        "expected_decision": "ESCALATE",
        "message": "I am locked out of my account because I no longer have access to my old phone for OTP."
    },
    # 17. TECHNICAL_AND_PLATFORM_ISSUES (Fire TV / Auto-Handle)
    {
        "id": "sample_17",
        "category": "TECHNICAL_AND_PLATFORM_ISSUES",
        "expected_decision": "AUTO_HANDLE",
        "message": "My Firestick is saying network error even though my Wi-Fi is working on my phone."
    },
    # 18. TECHNICAL_AND_PLATFORM_ISSUES (Checkout Error / Auto-Handle)
    {
        "id": "sample_18",
        "category": "TECHNICAL_AND_PLATFORM_ISSUES",
        "expected_decision": "AUTO_HANDLE",
        "message": "Website keeps throwing error 500 every time I click proceed to checkout in cart."
    },
    # 19. RETURNS_AND_EXCHANGES (Label / Printer / Auto-Handle)
    {
        "id": "sample_19",
        "category": "RETURNS_AND_EXCHANGES",
        "expected_decision": "AUTO_HANDLE",
        "message": "Need to return a shirt but I don't have a printer for the return shipping label."
    },
    # 20. RETURNS_AND_EXCHANGES (Pickup Delay / Escalate)
    {
        "id": "sample_20",
        "category": "RETURNS_AND_EXCHANGES",
        "expected_decision": "ESCALATE",
        "message": "Return pickup was scheduled 4 days ago and courier never showed up! When are they coming?"
    }
]

def run_verification(output_path: str = "data/analysis/pipeline_verification_results.json"):
    print("=" * 80)
    print("MILESTONE 11: FULL PIPELINE VERIFICATION SUITE (@AmazonHelp)")
    print("=" * 80)

    agent = HybridSupportAgent()
    results = []

    stage_successes = {
        "preprocessing": 0,
        "classification": 0,
        "retrieval": 0,
        "escalation_decision": 0,
        "response_generation": 0,
        "schema_conformance": 0
    }

    start_total = time.time()

    for idx, test_case in enumerate(SAMPLE_TEST_QUERIES, 1):
        sample_id = test_case["id"]
        msg = test_case["message"]
        cat = test_case["category"]

        print(f"\n[{idx:02d}/20] Testing {sample_id} ({cat})...")
        print(f"     Query: \"{msg[:75]}...\"")

        t0 = time.time()
        res = agent.process_message(msg, top_k=3, include_signals=True)
        latency = round(time.time() - t0, 3)

        # Stage Validations
        # 1. Preprocessing
        if res.get("intent") is not None:
            stage_successes["preprocessing"] += 1

        # 2. Classification
        if res.get("intent") != "UNKNOWN" and 0.0 <= res.get("intent_confidence", -1) <= 1.0:
            stage_successes["classification"] += 1

        # 3. Retrieval
        if isinstance(res.get("retrieved_cases"), list) and len(res["retrieved_cases"]) > 0:
            stage_successes["retrieval"] += 1

        # 4. Escalation Decision
        if res.get("decision") in ["AUTO_HANDLE", "ESCALATE"] and res.get("reason"):
            stage_successes["escalation_decision"] += 1

        # 5. Response Generation
        if res.get("reply") and len(res["reply"].strip()) > 10:
            stage_successes["response_generation"] += 1

        # 6. Schema Conformance
        required_keys = ["intent", "intent_confidence", "retrieved_cases", "reply", "decision", "reason"]
        if all(k in res for k in required_keys):
            stage_successes["schema_conformance"] += 1

        print(f"     -> Intent: {res['intent']} ({res['intent_confidence']*100:.1f}%)")
        print(f"     -> Decision: {res['decision']} | Latency: {latency:.2f}s")
        print(f"     -> Reply: \"{res['reply'][:85]}...\"")

        results.append({
            "test_case_id": sample_id,
            "category": cat,
            "input_message": msg,
            "latency_seconds": latency,
            "pipeline_output": res
        })

    total_duration = round(time.time() - start_total, 2)
    avg_latency = round(sum(r["latency_seconds"] for r in results) / len(results), 3)

    auto_handle_count = sum(1 for r in results if r["pipeline_output"]["decision"] == "AUTO_HANDLE")
    escalate_count = sum(1 for r in results if r["pipeline_output"]["decision"] == "ESCALATE")

    print("\n" + "=" * 80)
    print("PIPELINE VERIFICATION SUMMARY:")
    print("=" * 80)
    print(f"  Total Inquiries Processed:     {len(results)}")
    print(f"  Total Run Duration:            {total_duration}s (Avg: {avg_latency}s / inquiry)")
    print(f"  AUTO_HANDLE Decisions:         {auto_handle_count} ({auto_handle_count/len(results)*100:.1f}%)")
    print(f"  ESCALATE Decisions:            {escalate_count} ({escalate_count/len(results)*100:.1f}%)")
    print("-" * 80)
    print("  Stage Verification Rates:")
    for stage, count in stage_successes.items():
        pct = (count / len(results)) * 100
        print(f"    {stage:<24} : {count}/{len(results)} ({pct:.1f}%)")
    print("=" * 80)

    output_payload = {
        "milestone": "Milestone 11: Final Hybrid Support Agent",
        "verification_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_queries": len(results),
        "total_duration_seconds": total_duration,
        "average_latency_seconds": avg_latency,
        "stage_verification_rates": {
            k: round(v / len(results), 4) for k, v in stage_successes.items()
        },
        "decision_distribution": {
            "auto_handle_count": auto_handle_count,
            "escalate_count": escalate_count
        },
        "detailed_results": results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[SUCCESS] Pipeline verification saved to: {output_path}")
    return output_payload

if __name__ == "__main__":
    run_verification()
