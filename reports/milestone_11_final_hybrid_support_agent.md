# Milestone 11: Final Hybrid Support Agent Report (`AmazonHelp`)

**Generated On:** 2026-09-10 16:24:00  
**Target Brand:** `@AmazonHelp`  
**System Architecture:** Hybrid Modular Support Pipeline (Classical ML + Dense Vector Search + Deterministic Safety Guardrails + Structured LLM Generation)  
**Primary Interface:** [`src/pipeline.py`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/src/pipeline.py)  
**Automated Test Suite:** `tests/` (19 / 19 unit tests passing, 100% success)  
**Verification Suite:** [`src/evaluate_pipeline.py`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/src/evaluate_pipeline.py)  
**Verification Audit Results:** `data/analysis/pipeline_verification_results.json`  

---

## 1. Executive Summary

Milestone 11 unifies all previous milestones into a **production-ready, fully reproducible, modular customer support pipeline** for `@AmazonHelp`.

In accordance with strict enterprise requirements, the architecture avoids monolithic, opaque frameworks (no LangChain, no LlamaIndex, no complex agentic overhead). Instead, it employs a clean, decoupled 5-stage pipeline where each component performs one specific, auditable responsibility.

### Key Highlights:
1. **Unified Schema Conformance:** Every request produces a standardized, machine-readable JSON object:
   ```json
   {
     "intent": "ORDER_DELIVERY_AND_TRACKING",
     "intent_confidence": 0.9859,
     "retrieved_cases": [...],
     "reply": "I understand your concern about your parcel! Most couriers deliver until 9 PM...",
     "decision": "AUTO_HANDLE",
     "reason": "High intent confidence (0.99), strong historical grounding (sim: 0.83, precedents: 3), and zero safety risk flags detected."
   }
   ```
2. **Deterministic Safety Guardrails:** Prevents black-box hallucinations by routing safety-critical issues (account takeovers, legal threats, courier physical abuse, card fraud, and false delivery disputes) directly to specialized human workflows.
3. **Rigorous Test Coverage:** **19 automated unit tests** across 5 test suites (`tests/`), achieving a **100% pass rate**.
4. **End-to-End Pipeline Verification:** Validated on 20 diverse representative customer inquiries spanning all 10 intents and multiple difficulty tiers, achieving **100% verification** across all stage transitions with an average latency of **2.92s**.

---

## 2. Architecture & Data Flow

```
                      Incoming Customer Message
                                 │
                                 ▼
                     [Stage 1: Preprocessing]
                  (Unicode normalization, whitespace
                   collapse, control char cleanup)
                                 │
                                 ▼
                 [Stage 2: Intent Classification]
              (TF-IDF + Multinomial Logistic Regression:
               predicts intent, confidence, and margin)
                                 │
                                 ▼
                [Stage 3: Historical Precedent Retrieval]
             (Dense FAISS IndexFlatIP + all-MiniLM-L6-v2:
              retrieves top-k verified resolution precedents)
                                 │
                                 ▼
                   [Stage 4: Escalation Routing Policy]
              (Multi-signal evaluation: confidence, margin,
               similarity, evidence count, and 7 safety rules)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       Decision: `AUTO_HANDLE`         Decision: `ESCALATE`
                 │                               │
                 ▼                               ▼
     [Stage 5A: Grounded LLM Gen]     [Stage 5B: Empathetic Handoff]
    (Vertex AI gemini-2.5-flash:     (Immediate confidential transfer
     structured grounded reply)       message protecting customer PII)
                 │                               │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                    Final Standardized Output
             {"intent", "intent_confidence", "retrieved_cases",
              "reply", "decision", "reason"}
```

---

## 3. Modular Components Breakdown

| Component | Path | Responsibility | Technology Stack |
| :--- | :--- | :--- | :--- |
| **Preprocessing** | [`src/preprocessing.py`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/src/preprocessing.py) | Input text normalization, unicode NFKC normalization, whitespace collapsing, control character stripping. | Python `unicodedata`, `re` |
| **Classification** | [`src/predict_intent.py`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/src/predict_intent.py) | Predicts 1 of 10 customer intents with calibrated class probabilities and confidence margins. | `scikit-learn`, `joblib` |
| **Retrieval** | [`src/retrieve_resolutions.py`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/src/retrieve_resolutions.py) | Extracts top-$k$ nearest historical support cases from 12,000 resolved dialogue records. | `FAISS`, `sentence-transformers` |
| **Escalation** | [`src/escalation_policy.py`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/src/escalation_policy.py) | Evaluates intent confidence, margin, similarity, precedent counts, and safety rules to route to `AUTO_HANDLE` or `ESCALATE`. | Multi-Signal Rule Engine |
| **Generation** | [`src/generate_response.py`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/src/generate_response.py) | Generates grounded, empathetic, concise replies strictly adhering to historical precedent links and policies. | Google Vertex AI `gemini-2.5-flash` |
| **Pipeline** | [`src/pipeline.py`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/src/pipeline.py) | Unified orchestrator providing `HybridSupportAgent` and `run_support_agent(message)`. | Python standard pipeline |

---

## 4. Automated Unit Testing Suite Results

The unit test suite was executed via `pytest tests/ -v`:

```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\prakh\OneDrive\Desktop\Hiver
collected 19 items

tests/test_classification.py::test_model_artifact_exists PASSED          [  5%]
tests/test_classification.py::test_predict_intent_structure PASSED       [ 10%]
tests/test_classification.py::test_predict_intent_probabilities_sum_to_one PASSED [ 15%]
tests/test_classification.py::test_predict_intent_empty_text PASSED      [ 21%]
tests/test_escalation.py::test_detect_safety_risks PASSED                [ 26%]
tests/test_escalation.py::test_escalation_on_safety_trigger PASSED       [ 31%]
tests/test_escalation.py::test_auto_handle_on_clear_inquiry PASSED       [ 36%]
tests/test_pipeline.py::test_pipeline_output_schema_conformance PASSED   [ 42%]
tests/test_pipeline.py::test_pipeline_retrieved_case_structure PASSED    [ 47%]
tests/test_pipeline.py::test_pipeline_safety_escalation_flow PASSED      [ 52%]
tests/test_pipeline.py::test_pipeline_blank_message_handling PASSED      [ 57%]
tests/test_preprocessing.py::test_clean_customer_message_basic PASSED    [ 63%]
tests/test_preprocessing.py::test_clean_customer_message_whitespace_and_newlines PASSED [ 68%]
tests/test_preprocessing.py::test_clean_customer_message_null_and_empty PASSED [ 73%]
tests/test_preprocessing.py::test_clean_customer_message_control_chars PASSED [ 78%]
tests/test_preprocessing.py::test_clean_customer_message_preserves_emojis_and_casing PASSED [ 84%]
tests/test_retrieval.py::test_retrieval_artifacts_exist PASSED           [ 89%]
tests/test_retrieval.py::test_retriever_initialization PASSED            [ 94%]
tests/test_retrieval.py::test_retrieval_query_structure PASSED           [100%]

======================== 19 passed in 62.88s (0:01:02) ========================
```

---

## 5. End-to-End Pipeline Verification ($N = 20$)

Tested on 20 representative customer queries in [`src/evaluate_pipeline.py`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/src/evaluate_pipeline.py) ([`data/analysis/pipeline_verification_results.json`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/data/analysis/pipeline_verification_results.json)):

| Pipeline Stage Gate | Verification Criteria | Passed | Success Rate |
| :--- | :--- | :---: | :---: |
| **1. Preprocessing Gate** | Valid non-null string, normalized unicode. | 20 / 20 | **100.0%** |
| **2. Intent Classification Gate** | Valid taxonomy intent, calibrated confidence $\in [0, 1]$. | 20 / 20 | **100.0%** |
| **3. Historical Retrieval Gate** | Top-3 cases retrieved with valid cosine similarities. | 20 / 20 | **100.0%** |
| **4. Escalation Decision Gate** | Binary `AUTO_HANDLE` or `ESCALATE` with stated reason. | 20 / 20 | **100.0%** |
| **5. Response Generation Gate** | Non-empty, grounded, polite reply. | 20 / 20 | **100.0%** |
| **6. Schema Conformance Gate** | 100% adherence to specified 6-field output dictionary. | 20 / 20 | **100.0%** |

### Latency Performance:
- **Average End-to-End Latency:** **2.922s** per inquiry.
- **Escalated Inquiries:** **~0.04s** (instantaneous safety intercept before calling LLM, saving inference costs and eliminating latency).
- **Auto-Handled Inquiries:** **~5.1s** (includes dense embedding, FAISS search, and Vertex AI structured generation).

---

## 6. Real-World Case Studies

### 🌟 Case 1: Automated Order Tracking Inquiry (`sample_01`)
- **Customer Query:** `"Where is my parcel? The tracking status says out for delivery for 6 hours."`
- **Output:**
  ```json
  {
    "intent": "ORDER_DELIVERY_AND_TRACKING",
    "intent_confidence": 0.9859,
    "retrieved_cases": [
      {
        "conversation_id": "conv_1461510",
        "similarity_score": 0.8288,
        "intent": "ORDER_DELIVERY_AND_TRACKING",
        "customer_problem": "do you know where my parcel is?? Says Dispatched Wed 1st but its not even out for delivery and should be coming today!",
        "brand_response": "@326241 Hey, Emily! I totally understand your concern. Parcels can be delivered until 21:00- please keep us posted! ^BL"
      }
    ],
    "reply": "I understand your concern about your parcel! Most couriers deliver until 9 PM. Please keep an eye on the tracking, and let us know if it doesn't arrive by then.",
    "decision": "AUTO_HANDLE",
    "reason": "High intent confidence (0.99), strong historical grounding (sim: 0.83, precedents: 3), and zero safety risk flags detected."
  }
  ```

---

### 🛡️ Case 2: Instant Safety Escalation on Account Theft (`sample_15`)
- **Customer Query:** `"My seller account was hacked and unauthorized orders were placed! Please lock it now."`
- **Output:**
  ```json
  {
    "intent": "ACCOUNT_ACCESS_AND_SECURITY",
    "intent_confidence": 0.8431,
    "retrieved_cases": [
      {
        "conversation_id": "conv_954149",
        "similarity_score": 0.6389,
        "intent": "ACCOUNT_ACCESS_AND_SECURITY"
      }
    ],
    "reply": "We take account security very seriously. Because this involves confidential account verification, I am immediately escalating your inquiry to an Account Specialist. Please do not share sensitive passwords or personal details on this public forum.",
    "decision": "ESCALATE",
    "reason": "Safety risk trigger detected: [ACCOUNT_COMPROMISE] requiring immediate human intervention. | Intent 'ACCOUNT_ACCESS_AND_SECURITY' is a sensitive category requiring verified human handling."
  }
  ```

---

## 7. Operational Usage & CLI Guide

### 1. Python API:
```python
from src.pipeline import run_support_agent

result = run_support_agent("Where is my package? It was supposed to be delivered today.")
print(result["decision"])  # 'AUTO_HANDLE'
print(result["reply"])     # Grounded brand reply
```

### 2. Command-Line Interface (CLI):
```powershell
# Standard CLI Output
.venv\Scripts\python src/pipeline.py --message "How do I return a damaged book?"

# Structured JSON Output
.venv\Scripts\python src/pipeline.py --message "Where is my parcel?" --json
```

### 3. Run Automated Unit Tests:
```powershell
.venv\Scripts\python -m pytest tests/ -v
```

---

## 8. Complete Project Milestone Audit (M4 – M11)

| Milestone | Title | Key Artifact | Status |
| :--- | :--- | :--- | :---: |
| **M4** | Intent Discovery | [`INTENT_GUIDE.md`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/INTENT_GUIDE.md) | **Completed** |
| **M5** | Golden Evaluation Set | [`data/processed/golden_evaluation_set.jsonl`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/data/processed/golden_evaluation_set.jsonl) | **Completed** |
| **M6** | Trivial Baseline | [`reports/milestone_6_trivial_baseline.md`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/reports/milestone_6_trivial_baseline.md) | **Completed** |
| **M7** | Classical ML Baseline | [`models/intent_tfidf_logistic_regression.joblib`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/models/intent_tfidf_logistic_regression.joblib) | **Completed** |
| **M8** | Historical Resolution Retrieval | [`models/historical_knowledge_base.faiss`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/models/historical_knowledge_base.faiss) | **Completed** |
| **M9** | Grounded Response Generation | [`reports/milestone_9_grounded_response_generation.md`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/reports/milestone_9_grounded_response_generation.md) | **Completed** |
| **M10**| Auto-Handle vs Human Escalation | [`reports/milestone_10_auto_handle_vs_escalation.md`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/reports/milestone_10_auto_handle_vs_escalation.md) | **Completed** |
| **M11**| Final Hybrid Support Agent | [`src/pipeline.py`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/src/pipeline.py) | **Completed** |
