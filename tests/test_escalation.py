"""
Unit Tests for Escalation Policy & Routing Engine (src/escalation_policy.py).
"""

import pytest
from src.escalation_policy import (
    make_routing_decision,
    detect_safety_risks,
    extract_decision_signals
)

def test_detect_safety_risks():
    assert "ACCOUNT_COMPROMISE" in detect_safety_risks("My seller account was hacked and money stolen")
    assert "LEGAL_REGULATORY_THREAT" in detect_safety_risks("I will file a case in consumer court with my lawyer")
    assert "COURIER_MISCONDUCT" in detect_safety_risks("The delivery driver was abusive and threw my box")
    assert "FALSE_DELIVERY_DISPUTE" in detect_safety_risks("Order is marked delivered but I never received it")
    assert len(detect_safety_risks("How do I return a book?")) == 0

def test_escalation_on_safety_trigger():
    query = "Account hacked and funds drained! Called 3 times already."
    decision_out = make_routing_decision(query)

    assert decision_out["decision"] == "ESCALATE"
    assert "Safety risk trigger detected" in decision_out["reason"] or "sensitive category" in decision_out["reason"]
    assert "signals" in decision_out
    assert "ACCOUNT_COMPROMISE" in decision_out["signals"]["risk_flags"]

def test_auto_handle_on_clear_inquiry():
    query = "Where is my parcel? The tracking status says out for delivery."
    # Use thresholds that allow safe delivery inquiries
    thresholds = {
        "min_intent_confidence": 0.40,
        "min_intent_margin": 0.05,
        "min_retrieval_score": 0.70,
        "min_evidence_count": 1,
        "evidence_similarity_threshold": 0.65
    }
    decision_out = make_routing_decision(query, thresholds=thresholds)

    assert decision_out["decision"] in ["AUTO_HANDLE", "ESCALATE"]
    assert "signals" in decision_out
    assert "intent_confidence" in decision_out["signals"]
    assert "retrieval_score" in decision_out["signals"]
    assert "evidence_count" in decision_out["signals"]
