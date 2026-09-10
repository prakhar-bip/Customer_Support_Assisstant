# Customer Support Assistant (`@AmazonHelp`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Milestones%201--11%20Complete-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-19%2F19%20Passing%20(100%25)-brightgreen.svg)]()
[![Evaluated on](https://img.shields.io/badge/Golden%20Set-200%20Hand--Reviewed%20Cases-purple.svg)]()

An end-to-end, enterprise-grade hybrid customer support pipeline built on Twitter Customer Support (TWCS) data, specialized for `@AmazonHelp`.

---

## 📌 Project Architecture

The pipeline seamlessly unifies classical machine learning, dense semantic vector search, deterministic safety guardrails, and structured LLM response generation into a clean, reproducible workflow:

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

## 📊 Benchmark Highlights

Evaluated blindly on the quarantined **Golden Evaluation Set ($N = 200$)** with zero data leakage:

| Pipeline Component | Metric | Benchmark Score | Target Standard |
| :--- | :--- | :---: | :---: |
| **Milestone 6: Trivial Baseline** | Accuracy / Macro F1 | 14.00% / 2.46% | Lower bound |
| **Milestone 7: Classical ML Baseline** | Accuracy / Macro F1 | **90.50% / 91.05%** | $\ge 75.0\%$ |
| **Milestone 8: Historical FAISS Retrieval** | Top-3 Recall / MRR | **81.00% / 0.7289** | $\ge 70.0\%$ |
| **Milestone 9: Grounded Response Generation**| Evidence Citation / Conciseness | **100.0% / 100.0%** | $\ge 90.0\%$ |
| **Milestone 10: Escalation Policy** | Auto-Handle Precision / Esc Recall | **83.33% / 89.55%** | $\ge 80.0\%$ |
| **Milestone 11: End-to-End Pipeline** | Stage Verification / Unit Tests | **100.0% / 100.0%** | $100\%$ |

---

## 📂 Repository Structure

```
.
├── INTENT_GUIDE.md               # 10-intent taxonomy specification & boundary rules
├── LABELING_GUIDE.md             # Golden set annotation specification & triage rubric
├── README.md                     # Project documentation & operational guide
├── requirements.txt              # Environment dependencies
├── data/
│   ├── analysis/                 # Machine-readable evaluation metrics & JSON reports
│   │   ├── baseline_logistic_regression_metrics.json
│   │   ├── baseline_majority_metrics.json
│   │   ├── escalation_evaluation_metrics.json
│   │   ├── escalation_threshold_calibration.json
│   │   ├── escalation_tradeoff_comparison.json
│   │   ├── generation_evaluation_results.json
│   │   ├── intent_metrics.json
│   │   ├── pipeline_verification_results.json
│   │   └── retrieval_evaluation_metrics.json
│   └── processed/
│       ├── golden_evaluation_set.jsonl  # 200 hand-reviewed golden evaluation records
│       ├── golden_evaluation_set.csv    # Spreadsheet version of golden set
│       └── historical_knowledge_base.jsonl # 12,000 historical resolution records
├── models/
│   ├── historical_knowledge_base.faiss  # Exact cosine similarity FAISS vector index (18 MB)
│   └── intent_tfidf_logistic_regression.joblib  # Trained production model (1.2 MB)
├── reports/                      # Formal engineering reports for every milestone
│   ├── dataset_analysis.md
│   ├── milestone_3_conversation_reconstruction.md
│   ├── milestone_4_intent_discovery.md
│   ├── milestone_5_golden_evaluation_set.md
│   ├── milestone_6_trivial_baseline.md
│   ├── milestone_7_classical_ml_baseline.md
│   ├── milestone_8_historical_resolution_retrieval.md
│   ├── milestone_9_grounded_response_generation.md
│   ├── milestone_10_auto_handle_vs_escalation.md
│   └── milestone_11_final_hybrid_support_agent.md
├── src/                          # Production Python modules
│   ├── baseline_majority.py          # Milestone 6 Majority Baseline evaluator
│   ├── build_knowledge_base.py       # Milestone 8 12k-dialogue KB indexer
│   ├── create_golden_set.py          # Milestone 5 Stratified sampling and dataset assembly
│   ├── discover_intents.py           # Milestone 4 Corpus analysis and intent clustering
│   ├── escalation_policy.py          # Milestone 10 Multi-signal escalation routing policy
│   ├── evaluate_escalation.py        # Milestone 10 Golden set escalation benchmark
│   ├── evaluate_generation.py        # Milestone 9 20-sample grounded response evaluator
│   ├── evaluate_pipeline.py          # Milestone 11 End-to-end pipeline verification suite
│   ├── evaluate_retrieval.py         # Milestone 8 Golden set retrieval evaluator
│   ├── generate_response.py          # Milestone 9 Vertex AI structured response generator
│   ├── pipeline.py                   # Milestone 11 Unified Hybrid Support Agent orchestrator
│   ├── predict_intent.py             # Milestone 7 Classical ML inference engine
│   ├── preprocessing.py              # Milestone 11 Text normalization & cleaning module
│   ├── retrieve_resolutions.py       # Milestone 8 FAISS dense semantic retrieval engine
│   ├── train_classical_baseline.py   # Milestone 7 TF-IDF + Logistic Regression training
│   ├── tune_escalation_thresholds.py # Milestone 10 Validation threshold grid tuner
│   └── validate_golden_set.py        # Milestone 5 10-gate quality test suite
└── tests/                        # Automated unit tests (pytest)
    ├── test_classification.py
    ├── test_escalation.py
    ├── test_pipeline.py
    ├── test_preprocessing.py
    └── test_retrieval.py
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/prakhar-bip/Customer_Support_Assisstant.git
cd Customer_Support_Assisstant

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline (CLI)
Execute the unified hybrid support agent on any customer inquiry:
```bash
# Standard Output
python src/pipeline.py --message "Where is my parcel? The tracking status says out for delivery."

# Machine-Readable Structured JSON Output
python src/pipeline.py --message "Where is my parcel? The tracking status says out for delivery." --json
```

Sample JSON Output:
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

### 3. Python API Integration
```python
from src.pipeline import run_support_agent

result = run_support_agent("My seller account was hacked and money stolen!")
print(result["decision"])  # ESCALATE
print(result["reason"])    # Safety risk trigger detected: [ACCOUNT_COMPROMISE]...
print(result["reply"])     # Confidential handoff message
```

### 4. Run Automated Unit Tests
```bash
pytest tests/ -v
```

### 5. Run Full Pipeline Verification Suite
```bash
python src/evaluate_pipeline.py
```

---

## 📜 Milestones Summary

| Milestone | Title | Focus Area | Status | Deliverables |
| :--- | :--- | :--- | :---: | :--- |
| **M1 & M2** | Dataset Exploration & Subsetting | TWCS analysis & `@AmazonHelp` filtering | ✅ Complete | [`reports/dataset_analysis.md`](reports/dataset_analysis.md) |
| **M3** | Conversation Reconstruction | Multi-turn dialogue tree linking | ✅ Complete | [`reports/milestone_3_conversation_reconstruction.md`](reports/milestone_3_conversation_reconstruction.md) |
| **M4** | Intent Discovery | Clustering & 10-intent taxonomy | ✅ Complete | [`INTENT_GUIDE.md`](INTENT_GUIDE.md), [`reports/milestone_4_intent_discovery.md`](reports/milestone_4_intent_discovery.md) |
| **M5** | Golden Evaluation Set | 200 hand-reviewed ground truth cases | ✅ Complete | [`LABELING_GUIDE.md`](LABELING_GUIDE.md), [`data/processed/golden_evaluation_set.jsonl`](data/processed/golden_evaluation_set.jsonl) |
| **M6** | Trivial Baseline | Majority-class baseline (Macro F1: 2.46%) | ✅ Complete | [`src/baseline_majority.py`](src/baseline_majority.py), [`reports/milestone_6_trivial_baseline.md`](reports/milestone_6_trivial_baseline.md) |
| **M7** | Classical ML Baseline | TF-IDF + Logistic Regression (Macro F1: 91.05%) | ✅ Complete | [`src/predict_intent.py`](src/predict_intent.py), [`reports/milestone_7_classical_ml_baseline.md`](reports/milestone_7_classical_ml_baseline.md) |
| **M8** | Historical Retrieval | Dense FAISS index over 12k resolved cases | ✅ Complete | [`src/retrieve_resolutions.py`](src/retrieve_resolutions.py), [`reports/milestone_8_historical_resolution_retrieval.md`](reports/milestone_8_historical_resolution_retrieval.md) |
| **M9** | Grounded Generation | Vertex AI structured grounded generation | ✅ Complete | [`src/generate_response.py`](src/generate_response.py), [`reports/milestone_9_grounded_response_generation.md`](reports/milestone_9_grounded_response_generation.md) |
| **M10**| Escalation Routing | Multi-signal trustworthy auto-handle policy | ✅ Complete | [`src/escalation_policy.py`](src/escalation_policy.py), [`reports/milestone_10_auto_handle_vs_escalation.md`](reports/milestone_10_auto_handle_vs_escalation.md) |
| **M11**| Final Hybrid Agent | Unified pipeline, unit tests & verification | ✅ Complete | [`src/pipeline.py`](src/pipeline.py), [`reports/milestone_11_final_hybrid_support_agent.md`](reports/milestone_11_final_hybrid_support_agent.md) |

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
