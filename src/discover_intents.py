"""
Intent Discovery and Validation Pipeline for @AmazonHelp.

This script reproduces the data-driven customer support intent taxonomy discovery
from the cleaned AmazonHelp conversation dataset. It executes pattern validation,
computes empirical frequencies and overlap metrics, and outputs machine-readable
audit summaries.
"""

import json
import os
import re
import sys
from collections import Counter
import pandas as pd

DATA_PATH = "data/processed/amazonhelp_conversations_clean.csv"
OUTPUT_DIR = "data/analysis"
OUTPUT_METRICS = os.path.join(OUTPUT_DIR, "intent_metrics.json")

# 10 Data-Derived Intent Taxonomy Definition
TAXONOMY_PATTERNS = {
    "ORDER_DELIVERY_AND_TRACKING": (
        r'\b(where(\'s|\s+is)\s+(my\s+)?(order|package|delivery|parcel|item)|tracking|track\s+(my\s+)?(order|package)|'
        r'status\s+of\s+(my\s+)?(order|package)|out\s+for\s+delivery|in\s+transit|dispatch(ed)?|shipping\s+delay|'
        r'(delivery|package|order|shipment|item|parcel)\s+(delay|delayed|is\s+late|late)|delayed\s+(delivery|package|order|shipment)|'
        r'hasn\'t\s+arrived|haven\'t\s+received\s+(my\s+)?(order|package|item)|not\s+yet\s+arrived|'
        r'still\s+waiting\s+for\s+(my\s+)?(order|package)|missed\s+delivery|delivery\s+date|'
        r'expected\s+delivery|when\s+will\s+it\s+be\s+delivered|when\s+will\s+my\s+order\s+arrive|rescheduled\s+delivery|'
        r'marked\s+(as\s+)?delivered|says?\s+delivered|showing\s+delivered|status\s+shows\s+delivered|'
        r'delivered\s+but\s+(i\s+)?(haven\'t|have\s+not|not|never)\s+(received|got)|delivered\s+to\s+wrong|'
        r'handed\s+to\s+resident|package\s+missing|never\s+received\s+package)\b'
    ),
    "REFUND_STATUS_AND_DISPUTES": (
        r'\b(refund|refunded|money\s+back|credited|not\s+refunded|refund\s+status|refund\s+amount|refund\s+not\s+received|'
        r'haven\'t\s+got\s+refund|a-?to-?z\s+claim|a2z\s+claim|buyer\s+guarantee|balance\s+withheld|chargeback|'
        r'money\s+deducted|reversal|deducted\s+without|cashback\s+not\s+received|return\s+refund|where\s+is\s+my\s+refund)\b'
    ),
    "DAMAGED_DEFECTIVE_OR_WRONG_ITEM": (
        r'\b(damaged|broken|cracked|shattered|dented|defective|faulty|not\s+working|stopped\s+working|dead\s+on\s+arrival|'
        r'wrong\s+(item|product|order|model|color|size)|different\s+product|fake\s+product|counterfeit|'
        r'missing\s+(item|parts?|contents|accessories)|empty\s+box|used\s+item|seal\s+broken|tampered|leaking|scratched)\b'
    ),
    "PRIME_MEMBERSHIP_AND_DIGITAL": (
        r'\b(prime\s+membership|prime\s+subscription|charged\s+for\s+prime|prime\s+auto-?renew|cancel\s+prime|'
        r'prime\s+fee|prime\s+member|prime\s+video|prime\s+music|twitch\s+prime|prime\s+delivery\s+fee|prime\s+day\s+offer)\b'
    ),
    "ORDER_CANCELLATION": (
        r'\b(cancel\s+(the|this|my|an)?\s*(order|item|package)|how\s+to\s+cancel|want\s+to\s+cancel|cancell?ation|'
        r'cancelled|canceling|accidental(ly)?\s+order(ed)?|placed\s+order\s+by\s+mistake|cancel\s+request|'
        r'cannot\s+cancel|unable\s+to\s+cancel|cancellation\s+request)\b'
    ),
    "PAYMENT_BILLING_AND_PROMOS": (
        r'\b(payment\s+failed|charged\s+twice|double\s+charged|extra\s+charge|unauthorized\s+charge|bank\s+debited|'
        r'money\s+debited|card\s+declined|gift\s+card|promo\s+code|coupon\s+code|voucher|amazon\s+pay|pay\s+balance|'
        r'promotional\s+credit|billing\s+issue|payment\s+deducted|payment\s+processing)\b'
    ),
    "CUSTOMER_SERVICE_AND_COURIER_FEEDBACK": (
        r'\b(rude\s+(delivery|driver|agent|executive|courier)|driver\s+(threw|dropped|left)|worst\s+customer\s+service|'
        r'pathetic\s+service|disgusting\s+service|cheat(ing|ed)?\s+customers|worst\s+experience|poor\s+customer\s+service|'
        r'horrible\s+service|unprofessional\s+agent|terrible\s+service|harassment\s+by\s+delivery|disappointed\s+with\s+service)\b'
    ),
    "ACCOUNT_ACCESS_AND_SECURITY": (
        r'\b(login|log\s+in|sign\s+in|password\s+reset|cannot\s+access\s+account|locked\s+out|account\s+locked|'
        r'account\s+suspended|account\s+blocked|account\s+on\s+hold|hacked|suspicious\s+activity|unauthorized\s+login|'
        r'two-?factor|2fa|otp\s+not\s+received|verification\s+code|change\s+email|phishing|scam\s+call|seller\s+central\s+suspended)\b'
    ),
    "TECHNICAL_AND_PLATFORM_ISSUES": (
        r'\b(app\s+crash(es|ed|ing)?|website\s+down|site\s+not\s+working|glitch|bug|error\s+code|server\s+error|'
        r'page\s+not\s+loading|checkout\s+error|cart\s+empty|kindle\s+app|firestick|alexa\s+app|cannot\s+place\s+order)\b'
    ),
    "RETURNS_AND_EXCHANGES": (
        r'\b(return\s+(the|this|my|an)?\s*(item|product|order|package)|how\s+to\s+return|want\s+to\s+return|'
        r'return\s+policy|return\s+window|schedule\s+(a\s+)?return|return\s+pickup|courier\s+pickup|pickup\s+scheduled|'
        r'pickup\s+not\s+done|return\s+request|send\s+back|drop\s+off\s+return|return\s+label|replace(ment)?\s+request|'
        r'exchange\s+for\s+another|exchange\s+request|exchange\s+policy|exchange\s+my)\b'
    )
}

def run_discovery():
    if not os.path.exists(DATA_PATH):
        print(f"Error: dataset not found at {DATA_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    total_rows = len(df)
    print(f"Loaded {total_rows:,} clean conversations.")

    # Compile regexes
    compiled = {name: re.compile(pat, re.IGNORECASE) for name, pat in TAXONOMY_PATTERNS.items()}

    # Match each message
    print("Evaluating taxonomy patterns across conversations...")
    all_hits = []
    for text in df['customer_message'].fillna('').astype(str):
        hits = [name for name, pat in compiled.items() if pat.search(text)]
        all_hits.append(hits)

    df['matched_intents'] = all_hits
    df['match_count'] = df['matched_intents'].apply(len)

    # Compute statistics
    summary_stats = []
    for name in TAXONOMY_PATTERNS:
        total_cnt = sum(1 for hits in all_hits if name in hits)
        pure_cnt = sum(1 for hits in all_hits if hits == [name])
        summary_stats.append({
            "intent_name": name,
            "total_count": total_cnt,
            "total_percentage": round(total_cnt / total_rows * 100, 2),
            "pure_count": pure_cnt,
            "pure_percentage": round(pure_cnt / total_rows * 100, 2),
            "overlap_count": total_cnt - pure_cnt
        })

    # Sort by total count descending
    summary_stats.sort(key=lambda x: x['total_count'], reverse=True)

    # Overlap pair counts
    overlap_pairs = Counter()
    for hits in all_hits:
        if len(hits) > 1:
            for i in range(len(hits)):
                for j in range(i + 1, len(hits)):
                    pair = tuple(sorted([hits[i], hits[j]]))
                    overlap_pairs[pair] += 1

    top_overlaps = [
        {"intent_a": p[0], "intent_b": p[1], "overlap_count": c}
        for p, c in overlap_pairs.most_common(15)
    ]

    matched_total = sum(1 for hits in all_hits if len(hits) > 0)

    # Prepare audit output
    output_payload = {
        "dataset_total_conversations": total_rows,
        "matched_conversations": matched_total,
        "matched_percentage": round(matched_total / total_rows * 100, 2),
        "taxonomy_intents_count": len(TAXONOMY_PATTERNS),
        "intent_distributions": summary_stats,
        "top_overlaps": top_overlaps
    }

    with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nSuccessfully wrote metrics to {OUTPUT_METRICS}")
    print("\n" + "=" * 80)
    print(f"{'Intent Name':<40} | {'Total':>7} | {'Pct':>6}% | {'Pure':>7} | {'Overlap':>7}")
    print("-" * 80)
    for stat in summary_stats:
        print(
            f"{stat['intent_name']:<40} | "
            f"{stat['total_count']:>7,} | "
            f"{stat['total_percentage']:>6.2f}% | "
            f"{stat['pure_count']:>7,} | "
            f"{stat['overlap_count']:>7,}"
        )
    print("=" * 80)
    print(f"Total conversations categorized: {matched_total:,} ({matched_total/total_rows*100:.2f}%)")

if __name__ == "__main__":
    run_discovery()
