"""
Escalation Policy and Routing Engine for Customer Support Assistant (@AmazonHelp).

Implements an explicit, multi-signal decision policy to deterministically route
incoming messages to either:
  - 'AUTO_HANDLE': Safe for automated resolution based on high intent confidence,
                   strong historical precedent, and zero safety risk flags.
  - 'ESCALATE': Requires human agent intervention due to low confidence,
                high ambiguity, novel/unsupported situation, or safety-critical risks.

Guarantees:
  - Zero unconstrained LLM hallucinations.
  - Transparent, auditable reasons for every routing decision.
  - Measurable quantitative signals:
      * intent_confidence: calibrated classifier probability P(y_top1|x)
      * intent_margin: probability difference (P1 - P2) detecting ambiguity
      * retrieval_score: top-1 FAISS cosine similarity
      * evidence_count: count of high-similarity precedents (>= tau_sim) matching intent
      * risk_flags: safety-critical pattern triggers
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, Any, Optional, List

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.predict_intent import predict_intent
from src.retrieve_resolutions import ResolutionRetriever

# Default operating thresholds (calibrated on held-out validation data)
DEFAULT_THRESHOLDS = {
    "min_intent_confidence": 0.55,
    "min_intent_margin": 0.20,
    "min_retrieval_score": 0.65,
    "min_evidence_count": 1,
    "evidence_similarity_threshold": 0.65,
}

# Safety Risk Patterns requiring mandatory human escalation
SAFETY_PATTERNS = {
    "ACCOUNT_COMPROMISE": re.compile(
        r'\b(hacked|compromised|unauthorized access|scammed|fraud|someone used my account|phishing|stolen account)\b',
        re.I
    ),
    "LEGAL_REGULATORY_THREAT": re.compile(
        r'\b(lawyer|attorney|legal action|court|consumer forum|consumer court|police|fir|sue you|suing|police complaint)\b',
        re.I
    ),
    "COURIER_MISCONDUCT": re.compile(
        r'\b(driver.*(rude|abusive|threatened|assaulted|misbehaved|harass)|courier.*(stole|threw|dumped))\b',
        re.I
    ),
    "FINANCIAL_DISPUTE_CHARGEBACK": re.compile(
        r'\b(chargeback|bank dispute|disputed with bank|unauthorized charge|card fraud|a-?to-?z\s+claim|a\s+to\s+z\s+guarantee|guarantee\s+claim)\b',
        re.I
    ),
    "FALSE_DELIVERY_DISPUTE": re.compile(
        r'(\b(marked|says?|showing|status)\s+(as\s+)?delivered\b.*\b(not|never|hasn\'?t|haven\'?t|didn\'?t|where|arrived|received)\b|\b(not|never|hasn\'?t|haven\'?t|didn\'?t)\s+(delivered|received|arrived)\b.*\b(showing|says?|marked|status)\b)',
        re.I
    ),
    "HIGH_SEVERITY_PRODUCT_ISSUE": re.compile(
        r'\b(tampered|counterfeit|fake|wrong item|dead on arrival|empty box|missing item inside)\b',
        re.I
    ),
    "PII_LEAK_IN_PUBLIC": re.compile(
        r'(\b\d{3}-\d{7}-\d{7}\b|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',
        re.I
    )
}

# Mandatory Human Escalation Intents (categories that legally/operationally require humans)
MANDATORY_ESCALATION_INTENTS = {
    "CUSTOMER_SERVICE_AND_COURIER_FEEDBACK",
    "ACCOUNT_ACCESS_AND_SECURITY"
}

_retriever_instance = None

def get_retriever() -> ResolutionRetriever:
    """Lazy load and cache FAISS ResolutionRetriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = ResolutionRetriever()
    return _retriever_instance

def detect_safety_risks(text: str) -> List[str]:
    """Scan incoming customer message for safety-critical pattern triggers."""
    detected = []
    for risk_name, pattern in SAFETY_PATTERNS.items():
        if pattern.search(text):
            detected.append(risk_name)
    return detected

def extract_decision_signals(
    text: str,
    top_k: int = 5,
    evidence_sim_threshold: float = 0.65,
    retriever: Optional[ResolutionRetriever] = None
) -> Dict[str, Any]:
    """
    Extract all measurable signals required for the routing decision.

    Returns:
        dict: {
            "predicted_intent": str,
            "intent_confidence": float,
            "intent_margin": float,
            "top_probabilities": dict,
            "retrieval_score": float,
            "evidence_count": int,
            "retrieved_cases": list,
            "risk_flags": list,
            "is_mandatory_intent": bool
        }
    """
    # 1. Intent Classifier Signals
    intent_res = predict_intent(text)
    pred_intent = intent_res["predicted_intent"]
    top_conf = intent_res["confidence"]
    all_probs = intent_res.get("all_probabilities", {})

    prob_values = sorted(all_probs.values(), reverse=True)
    p1 = prob_values[0] if len(prob_values) > 0 else 0.0
    p2 = prob_values[1] if len(prob_values) > 1 else 0.0
    intent_margin = round(p1 - p2, 4)

    # 2. Retrieval Signals from Historical Knowledge Base
    if retriever is None:
        retriever = get_retriever()

    retrieved = retriever.retrieve(text, top_k=top_k)
    top_sim = round(retrieved[0].get("similarity_score", retrieved[0].get("similarity", 0.0)), 4) if retrieved else 0.0

    # Evidence count: precedents with sim >= threshold AND matching predicted intent
    matching_evidence = [
        c for c in retrieved
        if c.get("similarity_score", c.get("similarity", 0.0)) >= evidence_sim_threshold and c.get("intent") == pred_intent
    ]
    evidence_count = len(matching_evidence)

    # 3. Safety Risk Signals
    risk_flags = detect_safety_risks(text)
    is_mandatory_intent = pred_intent in MANDATORY_ESCALATION_INTENTS

    return {
        "predicted_intent": pred_intent,
        "intent_confidence": top_conf,
        "intent_margin": intent_margin,
        "top_probabilities": dict(list(all_probs.items())[:3]),
        "retrieval_score": top_sim,
        "evidence_count": evidence_count,
        "retrieved_cases": retrieved,
        "risk_flags": risk_flags,
        "is_mandatory_intent": is_mandatory_intent
    }

def make_routing_decision(
    text: str,
    thresholds: Optional[Dict[str, float]] = None,
    signals: Optional[Dict[str, Any]] = None,
    retriever: Optional[ResolutionRetriever] = None
) -> Dict[str, Any]:
    """
    Apply the explicit multi-signal decision policy to route customer query.

    Args:
        text (str): Customer message.
        thresholds (dict, optional): Custom thresholds dictionary.
        signals (dict, optional): Pre-computed signals dictionary to avoid recomputation.
        retriever (ResolutionRetriever, optional): Shared retriever instance.

    Returns:
        dict: Standardized output schema:
        {
          "decision": "AUTO_HANDLE" | "ESCALATE",
          "reason": "...",
          "signals": {
              "intent_confidence": float,
              "retrieval_score": float,
              "evidence_count": int,
              "intent_margin": float,
              "risk_flags": list
          }
        }
    """
    t = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        t.update(thresholds)

    if signals is None:
        signals = extract_decision_signals(
            text,
            top_k=5,
            evidence_sim_threshold=t["evidence_similarity_threshold"],
            retriever=retriever
        )

    # Signal variables
    pred_intent = signals["predicted_intent"]
    conf = signals["intent_confidence"]
    margin = signals["intent_margin"]
    sim = signals["retrieval_score"]
    ev_count = signals["evidence_count"]
    risk_flags = signals["risk_flags"]
    is_mandatory = signals["is_mandatory_intent"]

    # -------------------------------------------------------------
    # Explicit Decision Policy Evaluation
    # -------------------------------------------------------------
    escalate_reasons = []

    # Check 1: Deterministic Safety Risk Triggers
    if risk_flags:
        flags_str = ", ".join(risk_flags)
        escalate_reasons.append(
            f"Safety risk trigger detected: [{flags_str}] requiring immediate human intervention."
        )

    # Check 2: Mandatory Human Intent Categories
    if is_mandatory:
        escalate_reasons.append(
            f"Intent '{pred_intent}' is a sensitive category requiring verified human handling."
        )

    # Check 3: Intent Classification Ambiguity / Low Confidence
    if conf < t["min_intent_confidence"]:
        escalate_reasons.append(
            f"Low intent confidence ({conf:.4f} < {t['min_intent_confidence']:.2f})."
        )
    elif margin < t["min_intent_margin"]:
        escalate_reasons.append(
            f"High intent ambiguity (margin {margin:.4f} < {t['min_intent_margin']:.2f})."
        )

    # Check 4: Retrieval Precedent Coverage (Novel or unsupported situation)
    if sim < t["min_retrieval_score"]:
        escalate_reasons.append(
            f"Novel or unsupported situation: historical similarity ({sim:.4f} < {t['min_retrieval_score']:.2f})."
        )

    # Check 5: Evidence Grounding Density
    if ev_count < t["min_evidence_count"]:
        escalate_reasons.append(
            f"Insufficient verified historical precedents matching intent ({ev_count} < {t['min_evidence_count']})."
        )

    # -------------------------------------------------------------
    # Synthesize Final Routing Verdict
    # -------------------------------------------------------------
    if escalate_reasons:
        decision = "ESCALATE"
        reason = " | ".join(escalate_reasons)
    else:
        decision = "AUTO_HANDLE"
        reason = (
            f"High intent confidence ({conf:.2f}), strong historical grounding "
            f"(sim: {sim:.2f}, precedents: {ev_count}), and zero safety risk flags detected."
        )

    return {
        "decision": decision,
        "reason": reason,
        "signals": {
            "intent_confidence": conf,
            "retrieval_score": sim,
            "evidence_count": ev_count,
            "intent_margin": margin,
            "risk_flags": risk_flags
        },
        # Metadata for downstream pipeline inspection
        "metadata": {
            "predicted_intent": pred_intent,
            "is_mandatory_intent": is_mandatory,
            "top_probabilities": signals["top_probabilities"]
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-Signal Escalation Routing Engine (@AmazonHelp)")
    parser.add_argument("--text", type=str, required=True, help="Customer message text.")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format.")
    args = parser.parse_args()

    result = make_routing_decision(args.text)

    if args.json:
        # Output clean schema requested in specification
        clean_out = {
            "decision": result["decision"],
            "reason": result["reason"],
            "signals": result["signals"]
        }
        print(json.dumps(clean_out, indent=2))
    else:
        print("=" * 70)
        print(f"Customer Message: {args.text}")
        print(f"ROUTING DECISION: {result['decision']}")
        print(f"Stated Reason:    {result['reason']}")
        print("-" * 70)
        print("Measurable Signals:")
        for k, v in result["signals"].items():
            print(f"  {k:<20} : {v}")
        print("=" * 70)

if __name__ == "__main__":
    main()
