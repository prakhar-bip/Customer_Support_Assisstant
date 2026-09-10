# Milestone 7: Classical ML Baseline Report (`AmazonHelp`)

**Generated On:** 2026-09-10 15:19:00  
**Target Brand:** `@AmazonHelp`  
**Model Architecture:** `TfidfVectorizer(1-2 ngrams, 10k features) + LogisticRegression(balanced)`  
**Training Pool:** 14,529 Labeled Conversations (80/20 Stratified Split: 11,623 Train / 2,906 Val)  
**Evaluation Set:** `data/processed/golden_evaluation_set.jsonl` (200 Hand-Reviewed Ground Truth Samples)  
**Trained Model Artifact:** `models/intent_tfidf_logistic_regression.joblib`  
**Inference Module:** `src/predict_intent.py`  
**Machine-Readable Metrics:** `data/analysis/baseline_logistic_regression_metrics.json`  

---

## 1. Executive Summary

Milestone 7 establishes our first learned machine learning intent classification benchmark: a **TF-IDF Vectorizer + Multinomial Logistic Regression** pipeline. 

While Milestone 6 defined the empirical lower bound (Majority Baseline: 14.00% Accuracy, 2.46% Macro F1), Milestone 7 establishes a **strong, transparent, interpretable, and computationally lightweight classical ML benchmark**:
- **Golden Evaluation Accuracy:** **90.50%** (181 / 200 correct predictions).
- **Golden Evaluation Macro F1:** **91.05%** (an absolute improvement of **+88.59%** over the majority baseline).
- **Validation Accuracy (80/20 Split):** **93.50%** | **Validation Macro F1:** **92.77%**.

This establishes the formal performance standard that any future deep learning or generative AI / LLM system must surpass to justify added compute, latency, and operational cost.

---

## 2. Methodology & Architecture

```
[Customer Query Text]
         │
         ▼
[TfidfVectorizer]
  - n-gram range: (1, 2) [unigrams & bigrams]
  - vocabulary ceiling: 10,000 top features
  - sublinear_tf: True [logarithmic sublinear term frequency scaling 1 + log(tf)]
  - stop_words: 'english'
         │
         ▼
[Multinomial Logistic Regression]
  - solver: 'lbfgs' (multinomial cross-entropy loss)
  - regularization: C = 1.0 (L2 penalty)
  - class_weight: 'balanced' [penalizes under-represented class errors inversely to class prior]
  - max_iter: 1000
         │
         ▼
[Calibrated Probabilities across 10 Intents]
```

### Key Design Decisions:
1. **Balanced Class Weighting:** Training data volume ranges from 3,436 examples (`ORDER_DELIVERY_AND_TRACKING`) to 447 examples (`RETURNS_AND_EXCHANGES`). Setting `class_weight='balanced'` assigns weights $w_j = \frac{N}{K \cdot N_j}$, preventing dominant categories from suppressing minority intents.
2. **Sublinear TF Scaling:** Replacing raw term frequency with $1 + \log(\text{tf})$ dampens the influence of repeatedly repeated complaint words (e.g. *"worst worst worst"* or *"help help"*).
3. **Bigram Co-occurrence:** Bigrams capture critical compound phrases like `gift card`, `prime video`, `marked delivered`, and `dead on arrival` that define category boundaries.

---

## 3. Data Splitting & Data Leakage Prevention

To ensure absolute scientific validity:
- **Total Clean Dataset:** 79,665 conversations.
- **Golden Evaluation Set (Quarantined):** 200 conversations.
- **Data Leakage Overlap:** **0 conversations** ($\text{Train} \cap \text{Golden} = \emptyset$).
- **Training Pool:** 14,529 labeled conversations partitioned using an 80/20 stratified split:
  - **Training Set (80%):** 11,623 conversations.
  - **Validation Set (20%):** 2,906 conversations.

### Training Class Distribution:
1. `ORDER_DELIVERY_AND_TRACKING`: 3,436 (23.6%)
2. `REFUND_STATUS_AND_DISPUTES`: 2,565 (17.7%)
3. `DAMAGED_DEFECTIVE_OR_WRONG_ITEM`: 2,352 (16.2%)
4. `PRIME_MEMBERSHIP_AND_DIGITAL`: 1,684 (11.6%)
5. `ORDER_CANCELLATION`: 1,380 (9.5%)
6. `PAYMENT_BILLING_AND_PROMOS`: 874 (6.0%)
7. `CUSTOMER_SERVICE_AND_COURIER_FEEDBACK`: 663 (4.6%)
8. `ACCOUNT_ACCESS_AND_SECURITY`: 650 (4.5%)
9. `TECHNICAL_AND_PLATFORM_ISSUES`: 478 (3.3%)
10. `RETURNS_AND_EXCHANGES`: 447 (3.1%)

---

## 4. Evaluation Results on Golden Evaluation Set ($N = 200$)

The trained model was evaluated strictly and blindly on `data/processed/golden_evaluation_set.jsonl`:

| Metric | Score | Formulation / Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **90.50%** | 181 out of 200 hand-reviewed conversations correctly classified. |
| **Macro Precision** | **90.87%** | Mean precision across all 10 intent classes. |
| **Weighted Precision** | **90.90%** | Support-weighted precision across all 10 intent classes. |
| **Macro Recall** | **91.80%** | Mean recall across all 10 intent classes. |
| **Weighted Recall** | **90.50%** | Equivalent to overall accuracy. |
| **Macro F1-Score** | **91.05%** | Robust, unweighted harmonic mean across all 10 classes. |
| **Weighted F1-Score** | **90.43%** | Support-weighted harmonic mean across all 10 classes. |

---

## 5. Per-Intent Performance Breakdown

| Intent Code | Support | Predicted | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`ORDER_DELIVERY_AND_TRACKING`** | 28 | 25 | 96.00% | 85.71% | **90.57%** |
| **`REFUND_STATUS_AND_DISPUTES`** | 26 | 25 | 80.00% | 76.92% | **78.43%** |
| **`DAMAGED_DEFECTIVE_OR_WRONG_ITEM`** | 25 | 22 | 95.45% | 84.00% | **89.36%** |
| **`PRIME_MEMBERSHIP_AND_DIGITAL`** | 22 | 21 | 95.24% | 90.91% | **93.02%** |
| **`ORDER_CANCELLATION`** | 20 | 22 | 90.91% | 100.00% | **95.24%** |
| **`PAYMENT_BILLING_AND_PROMOS`** | 18 | 22 | 81.82% | 100.00% | **90.00%** |
| **`CUSTOMER_SERVICE_AND_COURIER_FEEDBACK`** | 16 | 15 | 100.00% | 93.75% | **96.77%** |
| **`ACCOUNT_ACCESS_AND_SECURITY`** | 15 | 14 | 92.86% | 86.67% | **89.66%** |
| **`TECHNICAL_AND_PLATFORM_ISSUES`** | 15 | 17 | 88.24% | 100.00% | **93.75%** |
| **`RETURNS_AND_EXCHANGES`** | 15 | 17 | 88.24% | 100.00% | **93.75%** |
| **Macro Average** | **200** | **200** | **90.87%** | **91.80%** | **91.05%** |

---

## 6. Confusion Matrix Analysis

```
Actual Ground Truth Class                 │ [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]
──────────────────────────────────────────┼─────────────────────────────────────────
[ 1] ORDER_DELIVERY_AND_TRACKING          │  24   3   1   0   0   0   0   0   0   0 
[ 2] REFUND_STATUS_AND_DISPUTES           │   0  20   0   1   0   3   0   0   1   1 
[ 3] DAMAGED_DEFECTIVE_OR_WRONG_ITEM      │   1   1  21   0   0   1   0   1   0   0 
[ 4] PRIME_MEMBERSHIP_AND_DIGITAL         │   0   1   0  20   1   0   0   0   0   0 
[ 5] ORDER_CANCELLATION                   │   0   0   0   0  20   0   0   0   0   0 
[ 6] PAYMENT_BILLING_AND_PROMOS           │   0   0   0   0   0  18   0   0   0   0 
[ 7] CUSTOMER_SERVICE_AND_COURIER_FEEDBACK│   0   0   0   0   0   0  15   0   0   1 
[ 8] ACCOUNT_ACCESS_AND_SECURITY          │   0   0   0   0   1   0   0  13   1   0 
[ 9] TECHNICAL_AND_PLATFORM_ISSUES        │   0   0   0   0   0   0   0   0  15   0 
[10] RETURNS_AND_EXCHANGES                │   0   0   0   0   0   0   0   0   0  15 
──────────────────────────────────────────┴─────────────────────────────────────────
Legend:
  [ 1] ORDER_DELIVERY_AND_TRACKING       [ 6] PAYMENT_BILLING_AND_PROMOS
  [ 2] REFUND_STATUS_AND_DISPUTES        [ 7] CUSTOMER_SERVICE_AND_COURIER_FEEDBACK
  [ 3] DAMAGED_DEFECTIVE_OR_WRONG_ITEM   [ 8] ACCOUNT_ACCESS_AND_SECURITY
  [ 4] PRIME_MEMBERSHIP_AND_DIGITAL      [ 9] TECHNICAL_AND_PLATFORM_ISSUES
  [ 5] ORDER_CANCELLATION                [10] RETURNS_AND_EXCHANGES
```

### Analysis of the 19 Misclassifications:
1. **Refund vs. Payment/Billing (3 cases):** Customers complaining that an unexpected transaction occurred or that cashback was not credited into their Amazon Pay balance. The model predicted `PAYMENT_BILLING_AND_PROMOS`, whereas ground truth labeled the missing money as `REFUND_STATUS_AND_DISPUTES`.
2. **Delivery Tracking vs. Refund (3 cases):** Customers whose delivery was overdue and who said *"cancel it and give my money back"*. Ground truth labeled the root cause as delivery delay (`ORDER_DELIVERY_AND_TRACKING`), while the linear model gave heavy weight to the term *"money back"*.
3. **Damaged Item vs. Delivery Tracking (1 case):** Transit damage where the customer remarked *"arrived completely crushed today"*. The model focused on *"arrived"* rather than *"crushed"*.
4. **Perfect Classifiers (100% Recall):** `ORDER_CANCELLATION`, `PAYMENT_BILLING_AND_PROMOS`, `TECHNICAL_AND_PLATFORM_ISSUES`, and `RETURNS_AND_EXCHANGES` achieved **100.00% recall**—every single ground truth instance was successfully detected.

---

## 7. Comparative Benchmark: Classical ML vs. Majority Baseline

| Metric | Majority Baseline (M6) | Classical ML (M7) | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 14.00% | **90.50%** | **+76.50%** |
| **Macro Precision** | 1.40% | **90.87%** | **+89.47%** |
| **Macro Recall** | 10.00% | **91.80%** | **+81.80%** |
| **Macro F1-Score** | 2.46% | **91.05%** | **+88.59%** |
| **Weighted F1-Score** | 3.44% | **90.43%** | **+86.99%** |

### Why This Comparison is Insightful:
The majority baseline demonstrated the catastrophic failure mode of zero-rule guessing (86% misrouting rate, 2.46% Macro F1). The classical TF-IDF + Logistic Regression model demonstrates that **n-gram term frequency features alone contain immense discriminative power** for customer support intents, raising Macro F1 from 2.46% to **91.05%** with sub-millisecond inference latency.

---

## 8. Reproducible Inference Module (`src/predict_intent.py`)

A production-ready inference function is provided in `src/predict_intent.py`:

```python
from src.predict_intent import predict_intent

result = predict_intent("My package is 3 days late and hasn't arrived, where is my order?")
print(result["predicted_intent"])
# 'ORDER_DELIVERY_AND_TRACKING'
print(result["confidence"])
# 0.9190 (91.90%)
```

### CLI Verification:
```bash
python src/predict_intent.py --text "I opened the box and the phone screen is completely cracked and shattered" --json
```
Output:
```json
{
  "text": "I opened the box and the phone screen is completely cracked and shattered",
  "predicted_intent": "DAMAGED_DEFECTIVE_OR_WRONG_ITEM",
  "confidence": 0.9191,
  "all_probabilities": {
    "DAMAGED_DEFECTIVE_OR_WRONG_ITEM": 0.9191,
    "TECHNICAL_AND_PLATFORM_ISSUES": 0.0215,
    "REFUND_STATUS_AND_DISPUTES": 0.0101,
    "ACCOUNT_ACCESS_AND_SECURITY": 0.01,
    "RETURNS_AND_EXCHANGES": 0.0082, ...
  }
}
```

---

## 9. Deliverables Summary

1. **`src/train_classical_baseline.py`:** Training and evaluation pipeline script.
2. **`src/predict_intent.py`:** Standalone reproducible inference module and CLI.
3. **`models/intent_tfidf_logistic_regression.joblib`:** Serialized production model artifact.
4. **`data/analysis/baseline_logistic_regression_metrics.json`:** Full machine-readable evaluation report.
5. **`reports/milestone_7_classical_ml_baseline.md`:** This technical report.
