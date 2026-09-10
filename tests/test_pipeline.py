"""
Unit Tests for Unified Hybrid Support Pipeline (src/pipeline.py).
"""

import pytest
from src.pipeline import HybridSupportAgent, run_support_agent

def test_pipeline_output_schema_conformance():
    query = "Where is my package? It was supposed to be delivered today."
    result = run_support_agent(query, top_k=3)

    # Required fields per Milestone 11 specification:
    required_keys = ["intent", "intent_confidence", "retrieved_cases", "reply", "decision", "reason"]
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"

    assert isinstance(result["intent"], str)
    assert isinstance(result["intent_confidence"], float)
    assert isinstance(result["retrieved_cases"], list)
    assert len(result["retrieved_cases"]) <= 3
    assert isinstance(result["reply"], str)
    assert len(result["reply"].strip()) > 0
    assert result["decision"] in ["AUTO_HANDLE", "ESCALATE"]
    assert isinstance(result["reason"], str)

def test_pipeline_retrieved_case_structure():
    query = "How do I return a broken coffee mug?"
    result = run_support_agent(query, top_k=2)

    assert len(result["retrieved_cases"]) > 0
    case = result["retrieved_cases"][0]
    assert "conversation_id" in case
    assert "similarity_score" in case
    assert "intent" in case
    assert "customer_problem" in case
    assert "brand_response" in case

def test_pipeline_safety_escalation_flow():
    query = "My seller account was hacked and all money stolen! Will take legal action."
    result = run_support_agent(query, top_k=3)

    assert result["decision"] == "ESCALATE"
    assert "reply" in result
    assert "account" in result["reply"].lower() or "escalat" in result["reply"].lower()

def test_pipeline_blank_message_handling():
    result = run_support_agent("")
    assert result["decision"] == "AUTO_HANDLE"
    assert "intent" in result
    assert len(result["reply"]) > 0
