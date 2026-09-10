"""
Final Hybrid Customer Support Agent Pipeline (@AmazonHelp).

Integrates all components into a single, modular, reproducible pipeline:
  Customer message
  → Preprocessing (src/preprocessing.py)
  → Intent Classification (src/predict_intent.py)
  → Historical Retrieval (src/retrieve_resolutions.py)
  → Escalation Decision (src/escalation_policy.py)
  → Historically Grounded Generation (src/generate_response.py)
  → Structured Output

Returns:
{
  "intent": "...",
  "intent_confidence": ...,
  "retrieved_cases": [...],
  "reply": "...",
  "decision": "AUTO_HANDLE | ESCALATE",
  "reason": "..."
}
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional, List

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import clean_customer_message
from src.predict_intent import predict_intent
from src.retrieve_resolutions import ResolutionRetriever
from src.escalation_policy import make_routing_decision, extract_decision_signals, DEFAULT_THRESHOLDS
from src.generate_response import generate_grounded_response

DEFAULT_CALIBRATION_PATH = "data/analysis/escalation_threshold_calibration.json"

class HybridSupportAgent:
    """
    Unified end-to-end hybrid support agent for @AmazonHelp.
    Combines classical ML, dense vector retrieval, deterministic safety guardrails,
    and structured LLM response generation.
    """

    def __init__(
        self,
        calibration_path: str = DEFAULT_CALIBRATION_PATH,
        custom_thresholds: Optional[Dict[str, float]] = None
    ):
        """Initialize and cache pipeline dependencies."""
        # 1. Load Calibrated Thresholds
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, "r", encoding="utf-8") as f:
                    calib_data = json.load(f)
                loaded_t = calib_data.get("selected_trustworthy_thresholds", {})
                self.thresholds.update(loaded_t)
            except Exception as e:
                print(f"[WARN] Could not load calibration from {calibration_path}: {e}")

        if custom_thresholds:
            self.thresholds.update(custom_thresholds)

        # 2. Lazy load / initialize retriever
        self.retriever = ResolutionRetriever()

    def process_message(
        self,
        customer_message: str,
        top_k: int = 3,
        include_signals: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the complete hybrid support pipeline on an incoming message.

        Args:
            customer_message (str): Raw incoming customer query.
            top_k (int): Number of historical precedents to retrieve.
            include_signals (bool): Include internal diagnostic signals in metadata.

        Returns:
            dict: Structured response schema:
            {
              "intent": str,
              "intent_confidence": float,
              "retrieved_cases": list,
              "reply": str,
              "decision": "AUTO_HANDLE" | "ESCALATE",
              "reason": str
            }
        """
        # Step 1: Preprocessing
        cleaned_text = clean_customer_message(customer_message)
        if not cleaned_text:
            return {
                "intent": "UNKNOWN",
                "intent_confidence": 0.0,
                "retrieved_cases": [],
                "reply": "Hello! How can we assist you with your Amazon order or account today?",
                "decision": "AUTO_HANDLE",
                "reason": "Empty or blank input received; prompted for customer inquiry."
            }

        # Step 2: Intent Classification & Signal Extraction
        # extract_decision_signals performs classification & retrieval in one shared step
        signals = extract_decision_signals(
            text=cleaned_text,
            top_k=top_k,
            evidence_sim_threshold=self.thresholds.get("evidence_similarity_threshold", 0.65),
            retriever=self.retriever
        )

        pred_intent = signals["predicted_intent"]
        intent_conf = signals["intent_confidence"]
        raw_retrieved = signals["retrieved_cases"]

        # Format retrieved cases cleanly for the response
        formatted_retrieved = []
        for case in raw_retrieved[:top_k]:
            formatted_retrieved.append({
                "conversation_id": case["conversation_id"],
                "similarity_score": case.get("similarity_score", 0.0),
                "intent": case.get("intent", "UNKNOWN"),
                "customer_problem": case.get("customer_problem", ""),
                "brand_response": case.get("brand_response", "")
            })

        # Step 3: Escalation Decision via Multi-Signal Policy
        routing_res = make_routing_decision(
            text=cleaned_text,
            thresholds=self.thresholds,
            signals=signals,
            retriever=self.retriever
        )
        decision = routing_res["decision"]  # "AUTO_HANDLE" or "ESCALATE"
        reason = routing_res["reason"]

        # Step 4: Historically Grounded Response Generation
        if decision == "AUTO_HANDLE":
            # Generate grounded self-service response via LLM
            try:
                gen_res = generate_grounded_response(
                    customer_message=cleaned_text,
                    top_k=top_k,
                    intent=pred_intent
                )
                reply = gen_res.reply
            except Exception as e:
                # Fallback in case of temporary LLM API unavailability
                reply = (
                    "We'd be glad to help with your inquiry! "
                    "Please reach out to our support team securely here: https://t.co/amazonhelp"
                )
        else:
            # Human Escalation Path: Empathetic handoff message protecting customer privacy
            risk_flags = signals.get("risk_flags", [])
            if "ACCOUNT_COMPROMISE" in risk_flags:
                reply = (
                    "We take account security very seriously. Because this involves confidential account verification, "
                    "I am immediately escalating your inquiry to an Account Specialist. Please do not share sensitive "
                    "passwords or personal details on this public forum."
                )
            elif "LEGAL_REGULATORY_THREAT" in risk_flags:
                reply = (
                    "Thank you for reaching out. We have logged your concern and escalated this matter to a senior specialist "
                    "who will review your dispute details and contact you directly."
                )
            elif "COURIER_MISCONDUCT" in risk_flags:
                reply = (
                    "We are truly sorry for your experience with the delivery service. We have escalated this courier report "
                    "to our logistics escalation team for investigation and immediate corrective action."
                )
            elif "FALSE_DELIVERY_DISPUTE" in risk_flags:
                reply = (
                    "I'm sorry your tracking shows delivered but you haven't received your package. Because this requires "
                    "a carrier trace, I am transferring your case to a human support agent to look into this for you right away."
                )
            else:
                reply = (
                    "To ensure your request is handled with full accuracy and care, I have escalated your issue to our "
                    "human support team. A representative will review the details and assist you shortly."
                )

        output_payload = {
            "intent": pred_intent,
            "intent_confidence": intent_conf,
            "retrieved_cases": formatted_retrieved,
            "reply": reply,
            "decision": decision,
            "reason": reason
        }

        if include_signals:
            output_payload["signals"] = routing_res["signals"]

        return output_payload

# Global singleton instance for fast reuse
_agent_singleton = None

def run_support_agent(message: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Convenience functional API to run the support agent.

    Args:
        message (str): Customer inquiry.
        top_k (int): Number of precedents to retrieve.

    Returns:
        dict: Standardized agent output schema.
    """
    global _agent_singleton
    if _agent_singleton is None:
        _agent_singleton = HybridSupportAgent()
    return _agent_singleton.process_message(message, top_k=top_k)

def main():
    parser = argparse.ArgumentParser(description="Hybrid Support Agent End-to-End Pipeline (@AmazonHelp)")
    parser.add_argument("--message", type=str, required=True, help="Incoming customer inquiry text.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of retrieved cases (default: 3).")
    parser.add_argument("--json", action="store_true", help="Output raw structured JSON format.")
    args = parser.parse_args()

    agent = HybridSupportAgent()
    result = agent.process_message(args.message, top_k=args.top_k)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 80)
        print("CUSTOMER INQUIRY:")
        print(f"  {args.message}")
        print("=" * 80)
        print(f"PREDICTED INTENT:    {result['intent']} (Confidence: {result['intent_confidence'] * 100:.2f}%)")
        print(f"ROUTING DECISION:    {result['decision']}")
        print(f"DECISION REASON:     {result['reason']}")
        print("-" * 80)
        print(f"HISTORICAL PRECEDENTS RETRIEVED ({len(result['retrieved_cases'])} cases):")
        for idx, case in enumerate(result['retrieved_cases'], 1):
            print(f"  [{idx}] Sim: {case['similarity_score']:.4f} | ID: {case['conversation_id']} | Intent: {case['intent']}")
        print("-" * 80)
        print("GENERATED REPLY:")
        print(f"  {result['reply']}")
        print("=" * 80)

if __name__ == "__main__":
    main()
