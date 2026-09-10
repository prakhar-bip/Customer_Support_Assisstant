# Milestone 6: Trivial Baseline Report (`AmazonHelp`)

**Generated On:** 2026-09-10 15:15:00  
**Target Brand:** `@AmazonHelp`  
**Model Type:** `MajorityClassClassifier` (Zero-Rule Empirical Prior Baseline)  
**Evaluation Set:** `data/processed/golden_evaluation_set.jsonl` (200 Hand-Reviewed Ground Truth Samples)  
**Pipeline Script:** `src/baseline_majority.py`  
**Machine-Readable Metrics:** `data/analysis/baseline_majority_metrics.json`  

---

## 1. Executive Summary & Baseline Objective

Milestone 6 establishes the first operational intent classification benchmark for `@AmazonHelp`: a **Majority-Class Baseline Classifier**. 

The goal of this baseline is not to achieve high performance, but rather to **establish the empirical lower bound**—the absolute floor that any heuristic rule engine, classical statistical classifier (e.g. TF-IDF + Logistic Regression / SVM / Naive Bayes), or deep learning / LLM system must decisively surpass.

### Core Rule:
$$\hat{y}_i = \arg\max_{c \in \mathcal{C}} P_{\text{train}}(y = c) = \text{"ORDER\_DELIVERY\_AND\_TRACKING"}$$

For every incoming customer query, regardless of vocabulary, context, or sentiment, the model blindly assigns the empirical mode of the training corpus: `ORDER_DELIVERY_AND_TRACKING`.

---

## 2. Zero Data Leakage Guarantee

To maintain strict scientific integrity, the majority class prior was estimated **strictly on the training pool**, completely isolated from the benchmark evaluation set:
- **Total Clean Dataset:** 79,665 conversations.
- **Golden Evaluation Set (Quarantined):** 200 conversations.
- **Isolated Training Pool:** 79,465 conversations ($79,665 - 200$).
- **Data Leakage Overlap:** **0 conversations** ($\text{Train} \cap \text{Golden} = \emptyset$).

In the training pool, `ORDER_DELIVERY_AND_TRACKING` represents the mode among actionable problem statements (accounting for ~4.63% total / 4.20% pure volume). Thus, the learned policy is:
$$\text{Predict } \hat{y} = \text{"ORDER\_DELIVERY\_AND\_TRACKING"} \quad \forall x$$

---

## 3. Global Evaluation Results on Golden Set ($N = 200$)

The majority-class classifier was evaluated blindly on the 200 hand-reviewed golden evaluation samples:

| Metric | Score | Formulation / Explanation |
| :--- | :---: | :--- |
| **Accuracy** | **14.00%** | $\frac{28 \text{ correct}}{200 \text{ total}}$ (Matches exact ground-truth prevalence of `ORDER_DELIVERY_AND_TRACKING`). |
| **Macro Precision** | **1.40%** | Unweighted average precision across all 10 intents ($\frac{14.0\% + 9 \times 0\%}{10}$). |
| **Weighted Precision** | **1.96%** | Precision weighted by class support ($0.14 \times 14.0\%$). |
| **Macro Recall** | **10.00%** | Unweighted average recall across all 10 intents ($\frac{100.0\% + 9 \times 0\%}{10}$). |
| **Weighted Recall** | **14.00%** | Recall weighted by class support (equivalent to accuracy). |
| **Macro F1-Score** | **2.46%** | Severe penalty for completely ignoring 9 out of 10 customer support categories. |
| **Weighted F1-Score** | **3.44%** | Weighted harmonic mean across support distribution. |

---

## 4. Per-Intent Performance Breakdown

Because the majority classifier assigns 100% of probability mass to a single intent, 9 of the 10 customer support categories receive **zero precision, zero recall, and zero F1-score**:

| Intent Code | Ground Truth Support | Predicted Count | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`ORDER_DELIVERY_AND_TRACKING`** | 28 | 200 | 14.00% | 100.00% | **24.56%** |
| `REFUND_STATUS_AND_DISPUTES` | 26 | 0 | 0.00% | 0.00% | 0.00% |
| `DAMAGED_DEFECTIVE_OR_WRONG_ITEM` | 25 | 0 | 0.00% | 0.00% | 0.00% |
| `PRIME_MEMBERSHIP_AND_DIGITAL` | 22 | 0 | 0.00% | 0.00% | 0.00% |
| `ORDER_CANCELLATION` | 20 | 0 | 0.00% | 0.00% | 0.00% |
| `PAYMENT_BILLING_AND_PROMOS` | 18 | 0 | 0.00% | 0.00% | 0.00% |
| `CUSTOMER_SERVICE_AND_COURIER_FEEDBACK` | 16 | 0 | 0.00% | 0.00% | 0.00% |
| `ACCOUNT_ACCESS_AND_SECURITY` | 15 | 0 | 0.00% | 0.00% | 0.00% |
| `TECHNICAL_AND_PLATFORM_ISSUES` | 15 | 0 | 0.00% | 0.00% | 0.00% |
| `RETURNS_AND_EXCHANGES` | 15 | 0 | 0.00% | 0.00% | 0.00% |
| **Total / Macro Average** | **200** | **200** | **1.40%** | **10.00%** | **2.46%** |

---

## 5. Confusion Matrix Analysis

The confusion matrix demonstrates the total operational collapse of the majority-class model. Every row represents actual ground truth, and every column represents model predictions:

```
Actual Ground Truth Class                 │ [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]
──────────────────────────────────────────┼─────────────────────────────────────────
[ 1] ORDER_DELIVERY_AND_TRACKING          │  28   0   0   0   0   0   0   0   0   0 
[ 2] REFUND_STATUS_AND_DISPUTES           │  26   0   0   0   0   0   0   0   0   0 
[ 3] DAMAGED_DEFECTIVE_OR_WRONG_ITEM      │  25   0   0   0   0   0   0   0   0   0 
[ 4] PRIME_MEMBERSHIP_AND_DIGITAL         │  22   0   0   0   0   0   0   0   0   0 
[ 5] ORDER_CANCELLATION                   │  20   0   0   0   0   0   0   0   0   0 
[ 6] PAYMENT_BILLING_AND_PROMOS           │  18   0   0   0   0   0   0   0   0   0 
[ 7] CUSTOMER_SERVICE_AND_COURIER_FEEDBACK│  16   0   0   0   0   0   0   0   0   0 
[ 8] ACCOUNT_ACCESS_AND_SECURITY          │  15   0   0   0   0   0   0   0   0   0 
[ 9] TECHNICAL_AND_PLATFORM_ISSUES        │  15   0   0   0   0   0   0   0   0   0 
[10] RETURNS_AND_EXCHANGES                │  15   0   0   0   0   0   0   0   0   0 
──────────────────────────────────────────┴─────────────────────────────────────────
Column Legend:
  [ 1]: ORDER_DELIVERY_AND_TRACKING (All 200 predictions concentrated here)
  [ 2]: REFUND_STATUS_AND_DISPUTES  (0 predictions)
  [ 3]: DAMAGED_DEFECTIVE_OR_WRONG_ITEM (0 predictions)
  [ 4]: PRIME_MEMBERSHIP_AND_DIGITAL (0 predictions)
  [ 5]: ORDER_CANCELLATION          (0 predictions)
  [ 6]: PAYMENT_BILLING_AND_PROMOS  (0 predictions)
  [ 7]: CUSTOMER_SERVICE_AND_COURIER_FEEDBACK (0 predictions)
  [ 8]: ACCOUNT_ACCESS_AND_SECURITY (0 predictions)
  [ 9]: TECHNICAL_AND_PLATFORM_ISSUES (0 predictions)
  [10]: RETURNS_AND_EXCHANGES       (0 predictions)
```

### Observations:
- **Zero Dispersion:** True positive count is 28 (on `ORDER_DELIVERY_AND_TRACKING`), while false positive count is **172**.
- **172 False Positives:** Every single ticket concerning broken items, stolen accounts, app crashes, or refund claims was misrouted as an order delivery status inquiry.

---

## 6. Analytical Discussion: Why This Baseline Matters

### 1. The Fallacy of Accuracy in Multi-Class NLP
If an engineer evaluates this model solely on accuracy, they might report: *"The baseline achieves 14.0% accuracy on a 10-class problem (outperforming random guessing at 10.0%)."*
However, this is completely deceptive:
- **Macro F1 is 2.46%:** Macro F1 calculates the unweighted mean of class F1 scores, exposing the fact that the model is entirely useless for 90% of customer intents.
- In multi-class customer support routing, **Macro F1 is the definitive metric of operational health**, because a model must perform reliably across rare and critical categories (such as account takeovers and payment failures), not just the most common fulfillment inquiries.

### 2. Operational Consequences in Production
Deploying a majority-class classifier in a real customer support environment would result in catastrophe:
- **86.0% Misrouting Rate:** 86 out of every 100 customer inquiries would be routed to the Logistics / Shipping Queue.
- **Account Security Disaster:** 100% of compromised account alerts, hacked seller credentials, and phishing reports would receive standard tracking links ("Check your order tracking at amazon.com/orders") rather than emergency security lockouts.
- **Customer Frustration Cascade:** A customer reporting a shattered laptop screen would be told when their parcel is arriving, triggering immediate customer rage and public brand embarrassment.

### 3. Purpose as a Scientific Benchmark
This trivial baseline establishes the **absolute minimum target**:
- Any viable heuristic or ML model must achieve:
  - $\text{Accuracy} \gg 14.0\%$
  - $\text{Macro F1} \gg 2.46\%$
- In Milestone 7 and beyond, our trained classifiers (TF-IDF + Logistic Regression/SVM, embeddings, or LLMs) will be benchmarked directly against this 2.46% Macro F1 baseline to quantify true semantic understanding.

---

## 7. Deliverables Summary

1. **`src/baseline_majority.py`:** Standalone, reproducible script executing the baseline pipeline and evaluation.
2. **`data/analysis/baseline_majority_metrics.json`:** Complete machine-readable audit report containing global metrics, per-class breakdown, and confusion matrix.
3. **`reports/milestone_6_trivial_baseline.md`:** This technical report.
