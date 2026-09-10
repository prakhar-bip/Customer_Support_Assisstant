"""
Milestone 8: Standalone Historical Resolution Retrieval Engine
--------------------------------------------------------------
Provides deterministic semantic retrieval of historically resolved customer-support
cases using sentence-transformers (all-MiniLM-L6-v2) and FAISS IndexFlatIP.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))


class ResolutionRetriever:
    """
    Retrieval engine for finding historically resolved customer-support cases
    using exact cosine similarity on dense embeddings.
    """

    def __init__(
        self,
        faiss_index_path: str = "models/historical_knowledge_base.faiss",
        metadata_path: str = "data/processed/historical_knowledge_base.jsonl",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(
                f"FAISS index file not found at: {faiss_index_path}. "
                "Please run `python src/build_knowledge_base.py` first."
            )
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found at: {metadata_path}. "
                "Please run `python src/build_knowledge_base.py` first."
            )

        # Load FAISS index
        self.index = faiss.read_index(faiss_index_path)
        self.total_vectors = self.index.ntotal

        # Load metadata records into memory mapped list
        self.metadata = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))

        if len(self.metadata) != self.total_vectors:
            raise ValueError(
                f"Count mismatch: FAISS index has {self.total_vectors} vectors, "
                f"but metadata file has {len(self.metadata)} records."
            )

        # Load sentence-transformers model
        self.model = SentenceTransformer(model_name)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        intent_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most similar historical cases for a given customer query.

        Args:
            query: Customer query / problem text.
            top_k: Number of historical cases to retrieve.
            min_similarity: Minimum cosine similarity threshold [0.0, 1.0].
            intent_filter: Optional intent code to restrict search.

        Returns:
            List of dicts containing similarity scores, conversation IDs,
            customer problems, brand resolutions, dialogue contexts, and intents.
        """
        if not query or not query.strip():
            return []

        # Encode query with unit L2 normalization
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # In IndexFlatIP with normalized vectors, inner product == cosine similarity
        fetch_k = top_k * 4 if intent_filter else top_k
        fetch_k = min(fetch_k, self.total_vectors)

        similarities, indices = self.index.search(query_embedding, fetch_k)

        results = []
        rank = 1

        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            sim_score = float(sim)
            if sim_score < min_similarity:
                continue

            case_meta = self.metadata[idx]

            # Optional intent filter
            if intent_filter and case_meta.get("intent") != intent_filter:
                continue

            result_item = {
                "rank": rank,
                "similarity_score": round(sim_score, 4),
                "conversation_id": case_meta["conversation_id"],
                "intent": case_meta.get("intent", "UNKNOWN"),
                "customer_problem": case_meta["customer_problem"],
                "brand_response": case_meta["brand_response"],
                "relevant_conversation_context": case_meta.get("relevant_conversation_context", ""),
                "resolution_status": case_meta.get("resolution_status", ""),
                "has_url": case_meta.get("has_url", False),
            }
            results.append(result_item)
            rank += 1

            if len(results) >= top_k:
                break

        return results


# Module-level singleton cache for fast repeated calls
_cached_retriever: Optional[ResolutionRetriever] = None


def retrieve_similar_cases(
    query_text: str,
    top_k: int = 5,
    min_similarity: float = 0.0,
    faiss_index_path: str = "models/historical_knowledge_base.faiss",
    metadata_path: str = "data/processed/historical_knowledge_base.jsonl",
) -> List[Dict[str, Any]]:
    """
    Convenience function for retrieving similar historical customer-support cases.
    Reuses a cached retriever instance across calls.
    """
    global _cached_retriever
    if _cached_retriever is None:
        _cached_retriever = ResolutionRetriever(
            faiss_index_path=faiss_index_path,
            metadata_path=metadata_path,
        )
    return _cached_retriever.retrieve(
        query=query_text,
        top_k=top_k,
        min_similarity=min_similarity,
    )


def main():
    parser = argparse.ArgumentParser(description="Retrieve historically similar customer-support cases.")
    parser.add_argument("--query", type=str, required=True, help="Customer inquiry text")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return")
    parser.add_argument("--min-similarity", type=float, default=0.0, help="Minimum similarity threshold")
    parser.add_argument("--intent", type=str, default=None, help="Filter by intent code")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    retriever = ResolutionRetriever()
    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        intent_filter=args.intent,
    )

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(f"\nQuery: \"{args.query}\"")
        print(f"Retrieved {len(results)} Historical Cases:\n")
        print("=" * 80)
        for res in results:
            print(f"Rank {res['rank']} | Similarity: {res['similarity_score']:.4f} | ID: {res['conversation_id']} | Intent: {res['intent']}")
            print(f"Customer Problem: {res['customer_problem']}")
            print(f"Historical Brand Response: {res['brand_response']}")
            print("-" * 80)


if __name__ == "__main__":
    main()
