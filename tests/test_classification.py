"""
Unit Tests for Intent Classification Inference Module (src/predict_intent.py).
"""

import pytest
import os
from src.predict_intent import predict_intent, get_model, DEFAULT_MODEL_PATH

def test_model_artifact_exists():
    assert os.path.exists(DEFAULT_MODEL_PATH), f"Model artifact missing at {DEFAULT_MODEL_PATH}"

def test_predict_intent_structure():
    query = "Where is my order? It was scheduled for delivery yesterday."
    res = predict_intent(query)

    assert "text" in res
    assert "predicted_intent" in res
    assert "confidence" in res
    assert "all_probabilities" in res

    assert res["predicted_intent"] == "ORDER_DELIVERY_AND_TRACKING"
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["all_probabilities"]) == 10

def test_predict_intent_probabilities_sum_to_one():
    query = "How do I cancel my prime membership subscription?"
    res = predict_intent(query)
    probs_sum = sum(res["all_probabilities"].values())
    assert abs(probs_sum - 1.0) < 0.01

def test_predict_intent_empty_text():
    res = predict_intent("")
    assert res["predicted_intent"] == "UNKNOWN"
    assert res["confidence"] == 0.0
