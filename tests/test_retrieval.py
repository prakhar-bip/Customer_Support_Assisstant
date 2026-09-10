"""
Unit Tests for Historical Precedent Retrieval Engine (src/retrieve_resolutions.py).
"""

import pytest
import os
from src.retrieve_resolutions import ResolutionRetriever, DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH

def test_retrieval_artifacts_exist():
    assert os.path.exists(DEFAULT_INDEX_PATH), f"FAISS index missing at {DEFAULT_INDEX_PATH}"
    assert os.path.exists(DEFAULT_METADATA_PATH), f"Metadata store missing at {DEFAULT_METADATA_PATH}"

def test_retriever_initialization():
    retriever = ResolutionRetriever()
    assert retriever.index is not None
    assert len(retriever.metadata) == 12000

def test_retrieval_query_structure():
    retriever = ResolutionRetriever()
    results = retriever.retrieve("I want my money back from Amazon Pay balance", top_k=3)

    assert len(results) == 3
    for idx, item in enumerate(results, 1):
        assert item["rank"] == idx
        assert "conversation_id" in item
        assert "similarity_score" in item
        assert "intent" in item
        assert "customer_problem" in item
        assert "brand_response" in item
        assert -1.0 <= item["similarity_score"] <= 1.0

    # Ensure results are sorted descending by similarity
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
