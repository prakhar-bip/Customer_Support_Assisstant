"""
Milestone 9: Historically Grounded Response Generation Module
-------------------------------------------------------------
Generates concise, brand-aligned customer support responses for @AmazonHelp,
grounded strictly in predicted intents and historically resolved support cases.

Supports:
- Google Vertex AI (using GCP_PROJECT_ID and GCP_LOCATION with gcloud credentials)
- Google AI Studio (fallback using GOOGLE_API_KEY)

Outputs structured JSON:
{
  "reply": "...",
  "reasoning_summary": "...",
  "evidence_ids": [...]
}
"""

import os
import sys
import json
import time
import subprocess
import argparse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

# Load environment variables
dotenv_path = find_dotenv()
if dotenv_path and os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

from google.oauth2.credentials import Credentials
from google import genai
from google.genai import types

from src.predict_intent import predict_intent
from src.retrieve_resolutions import retrieve_similar_cases, ResolutionRetriever


class GroundedResponse(BaseModel):
    """Structured response schema enforcing grounding and anti-hallucination."""

    reply: str = Field(
        ...,
        description=(
            "The concise, brand-aligned customer support reply to the customer. "
            "Must answer the inquiry, adhere to brand tone (@AmazonHelp), and strictly "
            "avoid inventing policies, dates, refund amounts, or unverified claims."
        ),
    )
    reasoning_summary: str = Field(
        ...,
        description=(
            "A concise, high-level rationale explaining how the historical evidence "
            "justifies or supports this reply. Do NOT expose internal step-by-step chain of thought."
        ),
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description=(
            "List of source conversation IDs from retrieved historical cases that directly "
            "ground or justify the policy, action, or guidance in this reply. "
            "Empty if retrieved cases were not relevant or applicable."
        ),
    )


def format_prompt(
    customer_message: str,
    predicted_intent: str,
    retrieved_cases: List[Dict[str, Any]],
) -> str:
    """
    Builds a prompt template clearly separating:
    1. Customer Input
    2. Retrieved Historical Evidence
    3. Operational Instructions & Constraints
    """
    evidence_blocks = []
    if retrieved_cases:
        for i, case in enumerate(retrieved_cases, start=1):
            block = (
                f"--- CASE {i} [ID: {case.get('conversation_id', 'N/A')}] ---\n"
                f"Similarity Score: {case.get('similarity_score', 0.0):.4f}\n"
                f"Historical Intent: {case.get('intent', 'UNKNOWN')}\n"
                f"Historical Customer Problem: {case.get('customer_problem', '').strip()}\n"
                f"Historical Brand Resolution: {case.get('brand_response', '').strip()}\n"
            )
            context = case.get("relevant_conversation_context", "")
            if context and len(context.strip()) > 0:
                context_lines = [line for line in context.strip().split("\n") if line.strip()][:4]
                block += f"Dialogue Context:\n" + "\n".join(context_lines) + "\n"
            evidence_blocks.append(block)
        formatted_evidence = "\n".join(evidence_blocks)
    else:
        formatted_evidence = "No historical cases retrieved."

    prompt = f"""You are an elite, highly reliable customer support assistant for @AmazonHelp on Twitter.
Your role is to formulate an accurate, empathetic, and concise reply to an incoming customer tweet.

================================================================================
SECTION 1: CUSTOMER INPUT
================================================================================
Customer Message:
"{customer_message}"

System Intent Classification:
{predicted_intent}

================================================================================
SECTION 2: RETRIEVED HISTORICAL EVIDENCE (HOW @AmazonHelp RESOLVED SIMILAR CASES)
================================================================================
{formatted_evidence}

================================================================================
SECTION 3: OPERATIONAL INSTRUCTIONS & CONSTRAINTS
================================================================================
1. GROUNDING IN HISTORICAL EVIDENCE:
   - Ground your answer in how @AmazonHelp historically addressed similar issues shown in SECTION 2.
   - If a retrieved case provides an official link (e.g. tracking link, carrier contact link, return portal link, or authenticated support link), adopt that resolution path.
   - Cite the corresponding source conversation ID(s) in `evidence_ids` ONLY for cases that directly supported your response.

2. STRICT ANTI-HALLUCINATION & POLICY INTEGRITY:
   - NEVER invent policies, compensation guarantees, exact delivery timeframes, or refund dates not supported by the evidence.
   - NEVER claim an order has shipped, is cancelled, or is refunded without authentication.
   - If the customer provided an order number or email publicly, politely advise them not to share personal order details on public social media (as @AmazonHelp standardly does).
   - If the retrieved cases DO NOT match or are unhelpful/irrelevant (e.g. low similarity or ambiguous query), do NOT force a citation. Instead, provide safe, brand-appropriate guidance or ask polite clarifying questions to diagnose their issue.

3. BRAND COMMUNICATION STYLE:
   - Tone: Empathetic, professional, calm, concise, and helpful (standard @AmazonHelp Twitter persona).
   - Style: Keep replies succinct (typically 1–3 sentences), suitable for customer support messaging.
   - Do NOT include internal agent sign-offs like '^XX' unless directly replicating a link format.

4. STRUCTURED OUTPUT REQUIREMENTS:
   - `reply`: The exact text to send to the customer.
   - `reasoning_summary`: A concise 1–2 sentence explanation of how the historical precedent justifies this answer. Do NOT reveal internal hidden chain of thought or reasoning steps.
   - `evidence_ids`: A list of conversation IDs from SECTION 2 that directly supported your reply.
"""
    return prompt


class ResponseGenerator:
    """
    Orchestrates Intent Prediction -> Historical Case Retrieval -> Structured LLM Generation.
    Supports Vertex AI and Google AI Studio modes.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        api_key: Optional[str] = None,
        faiss_index_path: str = "models/historical_knowledge_base.faiss",
        metadata_path: str = "data/processed/historical_knowledge_base.jsonl",
    ):
        self.model_name = model_name

        # Determine backend: Vertex AI vs AI Studio
        gcp_project = project_id or os.environ.get("GCP_PROJECT_ID")
        gcp_location = location or os.environ.get("GCP_LOCATION", "us-central1")

        self.backend = None
        self.client = None

        if gcp_project:
            try:
                # Attempt Vertex AI with gcloud access token
                token = subprocess.check_output(
                    "gcloud auth print-access-token", shell=True
                ).decode().strip()
                creds = Credentials(token)
                self.client = genai.Client(
                    vertexai=True,
                    project=gcp_project,
                    location=gcp_location,
                    credentials=creds,
                )
                self.backend = f"Vertex AI (project: {gcp_project}, location: {gcp_location})"
            except Exception as e:
                print(f"Warning: Could not initialize Vertex AI ({e}). Falling back to API key...")

        if self.client is None:
            key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise ValueError(
                    "Neither GCP_PROJECT_ID (with valid gcloud auth) nor GOOGLE_API_KEY was found."
                )
            self.client = genai.Client(api_key=key)
            self.backend = "Google AI Studio (API Key)"

        # Initialize FAISS Retriever
        self.retriever = ResolutionRetriever(
            faiss_index_path=faiss_index_path,
            metadata_path=metadata_path,
        )

    def generate(
        self,
        customer_message: str,
        top_k: int = 3,
        intent_override: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        """
        End-to-end response generation pipeline:
        1. Predict customer intent (if not provided).
        2. Retrieve top-k historical support cases.
        3. Formulate structured prompt.
        4. Generate structured response with Gemini 2.5 Flash on Vertex AI.
        """
        # Step 1: Intent Classification
        if intent_override:
            intent = intent_override
            intent_conf = 1.0
        else:
            intent_result = predict_intent(customer_message)
            intent = intent_result["predicted_intent"]
            intent_conf = intent_result.get("confidence", 0.0)

        # Step 2: Dense Historical Retrieval
        retrieved_cases = self.retriever.retrieve(
            query=customer_message,
            top_k=top_k,
            min_similarity=min_similarity,
        )

        # Step 3: Build Structured Prompt
        prompt = format_prompt(
            customer_message=customer_message,
            predicted_intent=intent,
            retrieved_cases=retrieved_cases,
        )

        # Step 4: Call LLM with Structured Schema (with retry logic)
        response = None
        last_exc = None
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GroundedResponse,
                        temperature=0.2,
                    ),
                )
                break
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    wait_time = 2 ** (attempt + 1)
                    time.sleep(wait_time)

        if response is None:
            raise RuntimeError(f"LLM generation failed after 3 attempts: {last_exc}") from last_exc

        # Step 5: Parse Structured JSON
        raw_text = response.text.strip()
        parsed = json.loads(raw_text)

        return {
            "customer_message": customer_message,
            "predicted_intent": intent,
            "intent_confidence": round(intent_conf, 4),
            "retrieved_cases_count": len(retrieved_cases),
            "retrieved_cases": retrieved_cases,
            "reply": parsed.get("reply", ""),
            "reasoning_summary": parsed.get("reasoning_summary", ""),
            "evidence_ids": parsed.get("evidence_ids", []),
            "backend": self.backend,
        }


_cached_response_generator: Optional[ResponseGenerator] = None

def generate_grounded_response(
    customer_message: str,
    top_k: int = 3,
    intent: Optional[str] = None
) -> GroundedResponse:
    """
    Convenience function to generate a grounded response using cached ResponseGenerator.
    """
    global _cached_response_generator
    if _cached_response_generator is None:
        _cached_response_generator = ResponseGenerator()
    res = _cached_response_generator.generate(
        customer_message=customer_message,
        top_k=top_k,
        intent_override=intent
    )
    return GroundedResponse(
        reply=res["reply"],
        reasoning_summary=res["reasoning_summary"],
        evidence_ids=res["evidence_ids"]
    )


def main():
    parser = argparse.ArgumentParser(description="Historically Grounded Customer Support Response Generator")
    parser.add_argument("--query", type=str, required=True, help="Customer message / tweet")
    parser.add_argument("--top-k", type=int, default=3, help="Number of historical cases to retrieve")
    parser.add_argument("--intent", type=str, default=None, help="Explicit intent override (optional)")
    parser.add_argument("--json", action="store_true", help="Output full JSON result")
    args = parser.parse_args()

    generator = ResponseGenerator()
    print(f"Active Backend: {generator.backend}", file=sys.stderr)
    result = generator.generate(
        customer_message=args.query,
        top_k=args.top_k,
        intent_override=args.intent,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 80)
        print("CUSTOMER INQUIRY:")
        print(f"  \"{result['customer_message']}\"")
        print(f"Predicted Intent: {result['predicted_intent']} (Conf: {result['intent_confidence']:.2%})")
        print(f"Retrieved {result['retrieved_cases_count']} historical cases.")
        print("-" * 80)
        print("GENERATED GROUNDED REPLY:")
        print(f"  {result['reply']}")
        print("-" * 80)
        print(f"Reasoning Summary: {result['reasoning_summary']}")
        print(f"Cited Evidence IDs: {result['evidence_ids']}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
