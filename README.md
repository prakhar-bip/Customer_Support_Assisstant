# Customer Support Assistant (`@AmazonHelp`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Milestones%201--7%20Complete-success.svg)]()
[![Evaluated on](https://img.shields.io/badge/Golden%20Set-200%20Hand--Reviewed%20Cases-purple.svg)]()

An end-to-end intelligent customer support classification and routing system built on Twitter Customer Support (TWCS) data, specialized for `@AmazonHelp`.

---

## 📌 Project Overview

This repository develops an enterprise-grade customer support assistant through systematic engineering milestones:
1. **Exploratory Data Analysis:** Profiled 2.8M customer support tweets and isolated 127K `@AmazonHelp` records.
2. **Conversation Thread Reconstruction:** Linked customer root tweets and agent replies into 79,665 clean multi-turn dialogue trees.
3. **Data-Driven Intent Discovery:** Empirically clustered customer problem spaces to formulate a robust 10-intent taxonomy ([`INTENT_GUIDE.md`](INTENT_GUIDE.md)).
4. **Golden Evaluation Benchmark:** Hand-reviewed and curated a quarantined 200-sample evaluation dataset stratified across intents, difficulty tiers, turn lengths, and escalation criteria ([`LABELING_GUIDE.md`](LABELING_GUIDE.md)).
5. **Baseline Benchmarks:**
   - **Milestone 6 (Trivial Majority Baseline):** Lower bound established at 14.00% Accuracy and 2.46% Macro F1.
   - **Milestone 7 (Classical ML Baseline):** TF-IDF + Logistic Regression pipeline achieving **90.50% Accuracy** and **91.05% Macro F1** on the quarantined golden evaluation set.

---

## 📊 Benchmark Results

Evaluated blindly on the quarantined **Golden Evaluation Set ($N = 200$)** with zero data leakage:

| Metric | Majority Baseline (M6) | Classical ML Baseline (M7) | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 14.00% | **90.50%** | **+76.50%** |
| **Macro Precision** | 1.40% | **90.87%** | **+89.47%** |
| **Macro Recall** | 10.00% | **91.80%** | **+81.80%** |
| **Macro F1-Score** | 2.46% | **91.05%** | **+88.59%** |
| **Weighted F1-Score** | 3.44% | **90.43%** | **+86.99%** |

### Per-Intent Performance Breakdown (Classical ML)

| Intent Code | Support | Predicted | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `ORDER_DELIVERY_AND_TRACKING` | 28 | 25 | 96.00% | 85.71% | **90.57%** |
| `REFUND_STATUS_AND_DISPUTES` | 26 | 25 | 80.00% | 76.92% | **78.43%** |
| `DAMAGED_DEFECTIVE_OR_WRONG_ITEM` | 25 | 22 | 95.45% | 84.00% | **89.36%** |
| `PRIME_MEMBERSHIP_AND_DIGITAL` | 22 | 21 | 95.24% | 90.91% | **93.02%** |
| `ORDER_CANCELLATION` | 20 | 22 | 90.91% | 100.00% | **95.24%** |
| `PAYMENT_BILLING_AND_PROMOS` | 18 | 22 | 81.82% | 100.00% | **90.00%** |
| `CUSTOMER_SERVICE_AND_COURIER_FEEDBACK` | 16 | 15 | 100.00% | 93.75% | **96.77%** |
| `ACCOUNT_ACCESS_AND_SECURITY` | 15 | 14 | 92.86% | 86.67% | **89.66%** |
| `TECHNICAL_AND_PLATFORM_ISSUES` | 15 | 17 | 88.24% | 100.00% | **93.75%** |
| `RETURNS_AND_EXCHANGES` | 15 | 17 | 88.24% | 100.00% | **93.75%** |
| **Macro Average** | **200** | **200** | **90.87%** | **91.80%** | **91.05%** |

---

## 📂 Repository Structure

```
.
├── INTENT_GUIDE.md               # 10-intent taxonomy specification & boundary rules
├── LABELING_GUIDE.md             # Golden set annotation specification & triage rubric
├── README.md                     # Project documentation & benchmark overview
├── requirements.txt              # Environment dependencies
├── data/
│   ├── analysis/                 # Machine-readable evaluation metrics & JSON reports
│   │   ├── baseline_logistic_regression_metrics.json
│   │   ├── baseline_majority_metrics.json
│   │   └── intent_metrics.json
│   └── processed/
│       ├── golden_evaluation_set.jsonl  # 200 hand-reviewed evaluation records
│       └── golden_evaluation_set.csv    # Spreadsheet version of golden set
├── models/
│   └── intent_tfidf_logistic_regression.joblib  # Trained production model (1.2 MB)
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── reports/                      # Detailed milestone engineering reports
│   ├── dataset_analysis.md
│   ├── milestone_3_conversation_reconstruction.md
│   ├── milestone_4_intent_discovery.md
│   ├── milestone_5_golden_evaluation_set.md
│   ├── milestone_6_trivial_baseline.md
│   └── milestone_7_classical_ml_baseline.md
└── src/                          # Production Python modules
    ├── baseline_majority.py          # Milestone 6 Majority Baseline evaluator
    ├── create_golden_set.py          # Stratified sampling and dataset assembly
    ├── discover_intents.py           # Corpus frequency analysis and taxonomy matcher
    ├── explore_dataset.py            # Initial TWCS dataset exploration
    ├── extract_brand_subset.py       # Brand isolation script
    ├── predict_intent.py             # Production CLI and Python inference API
    ├── reconstruct_conversations.py  # Graph-based conversation thread rebuilder
    ├── train_classical_baseline.py   # TF-IDF + Logistic Regression training pipeline
    └── validate_golden_set.py        # 10-gate automated golden set validation suite
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

### 2. Run Instant Inference
Use the pre-trained production model (`models/intent_tfidf_logistic_regression.joblib`) via CLI:
```bash
python src/predict_intent.py --text "My package is 3 days late and hasn't arrived, where is my order?" --json
```
Output:
```json
{
  "text": "My package is 3 days late and hasn't arrived, where is my order?",
  "predicted_intent": "ORDER_DELIVERY_AND_TRACKING",
  "confidence": 0.919,
  "all_probabilities": {
    "ORDER_DELIVERY_AND_TRACKING": 0.919,
    "REFUND_STATUS_AND_DISPUTES": 0.0245,
    "CUSTOMER_SERVICE_AND_COURIER_FEEDBACK": 0.0152, ...
  }
}
```

Or invoke in Python:
```python
from src.predict_intent import predict_intent

result = predict_intent("I received a broken laptop screen!")
print(result["predicted_intent"])  # DAMAGED_DEFECTIVE_OR_WRONG_ITEM
print(result["confidence"])        # 0.919
```

### 3. Validate the Golden Evaluation Set
Run the 10-gate quality test suite on the quarantined 200-sample benchmark:
```bash
python src/validate_golden_set.py
```

### 4. Train & Evaluate Baselines
Run the zero-rule majority-class baseline:
```bash
python src/baseline_majority.py
```

Train and evaluate the classical ML pipeline (80/20 train/val split on 14.5k records, tested on golden set):
```bash
python src/train_classical_baseline.py
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
| **M8** | Dense Semantic / Neural Model | Embeddings & Transformer classification | 🔄 Planned | In progress |
| **M9** | Automated Resolution & Routing | Action routing & safety guardrails | 🔄 Planned | Upcoming |

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
