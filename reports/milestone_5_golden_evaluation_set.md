# Milestone 5: Golden Evaluation Set Report (`AmazonHelp`)

**Generated On:** 2026-09-10 15:13:00  
**Target Brand:** `@AmazonHelp`  
**Dataset Reference:** `data/processed/golden_evaluation_set.jsonl` / `data/processed/golden_evaluation_set.csv`  
**Total Sample Size:** 200 Hand-Reviewed Benchmark Conversations  
**Validation Suite:** `src/validate_golden_set.py`  
**Labeling Manual:** `LABELING_GUIDE.md`  

---

## 1. Executive Summary

Milestone 5 establishes a gold-standard, human-reviewed evaluation set of **200 customer support conversations** for `@AmazonHelp`. 

In strict compliance with the assignment specification:
- **No Pseudo-Labeling:** The dataset was constructed through deterministic candidate extraction followed by manual inspection and ground-truth verification of customer messages, conversation context, resolution outcomes, and brand responses.
- **Statistical Stratification:** Rather than taking an arbitrary naive slice (e.g. the first 200 rows), the evaluation set was constructed via a **multi-dimensional stratified sampling strategy** covering all 10 finalized customer support intents, three difficulty tiers, varied conversation lengths, multiple time periods, and operational triage recommendations.
- **Reserved for Evaluation:** The golden evaluation set is strictly quarantined for blind evaluation and must **never** be used for model training, prompt few-shot tuning, or hyperparameter optimization.

---

## 2. Multi-Dimensional Stratified Sampling Strategy

The sampling framework was designed across five orthogonal dimensions:

```
                                  [200 Golden Samples]
                                           │
       ┌──────────────────┬────────────────┼─────────────────┬──────────────────┐
       ▼                  ▼                ▼                 ▼                  ▼
[10 Intent Classes] [3 Difficulty Tiers] [Turn Lengths] [Temporal Spread] [Operational Triage]
  28 Delivery (14%)   90 Easy (45%)        90 Short (45%)  Oct 2017 (Q4)     133 Auto-Handle (66.5%)
  26 Refund (13%)     70 Medium (35%)      80 Med (40%)    Nov 2017 (BF/CM)   67 Escalate (33.5%)
  25 Damaged (12.5%)  40 Hard (20%)        30 Long (15%)   Dec 2017 (Holiday)
  22 Prime (11%)                                           Early 2017 / Historic
  20 Cancel (10%)
  18 Payment (9%)
  16 Service (8%)
  15 Account (7.5%)
  15 Tech (7.5%)
  15 Return (7.5%)
```

### 1. Intent Distribution (Common vs. Rare)
The sampling ensures proportional representation of high-frequency operational drivers while guaranteeing statistically robust sample sizes for low-frequency and specialized categories:
- **High-Volume Logistics & Finance (39.5%):** `ORDER_DELIVERY_AND_TRACKING` (28, 14.0%), `REFUND_STATUS_AND_DISPUTES` (26, 13.0%), `DAMAGED_DEFECTIVE_OR_WRONG_ITEM` (25, 12.5%).
- **Mid-Volume Lifecycle & Services (30.0%):** `PRIME_MEMBERSHIP_AND_DIGITAL` (22, 11.0%), `ORDER_CANCELLATION` (20, 10.0%), `PAYMENT_BILLING_AND_PROMOS` (18, 9.0%).
- **Specialized & Low-Volume Support (30.5%):** `CUSTOMER_SERVICE_AND_COURIER_FEEDBACK` (16, 8.0%), `ACCOUNT_ACCESS_AND_SECURITY` (15, 7.5%), `TECHNICAL_AND_PLATFORM_ISSUES` (15, 7.5%), `RETURNS_AND_EXCHANGES` (15, 7.5%).
- **Class Balance Guarantee:** Every class contains at least 15 examples (well above the minimum threshold of 12).

### 2. Difficulty Tiers
- **`easy` (90 samples, 45.0%):** Prototypical customer requests with explicit, unambiguous keyword/syntactic triggers (e.g., *"where is my package I've waited since 7am"*, *"Kindle app keeps crashing on my Kindle HD"*).
- **`medium` (70 samples, 35.0%):** Multi-sentence inquiries with conversational context, customer frustration, informal language, or follow-ups across extended threads (e.g., *"5 chats, 1 call, 1 month later, issue still stands unresolved. But was promised an ETA of 1-2 days everytime"*).
- **`hard` (40 samples, 20.0%):** Complex edge cases with multi-intent overlaps, subtle linguistic nuances, or competing problem statements requiring strict application of the precedence hierarchy (e.g., damaged items where replacement is blocked, accidental orders where return is rejected, courier delivery rescheduling framed as emotional heartbreak).

### 3. Conversation Length Diversity
- **Short Interactions (Turn count = 2):** 90 samples (45.0%) — Prototypical single-turn customer query and brand reply.
- **Medium Interactions (Turn count = 3–5):** 80 samples (40.0%) — Multi-turn diagnostic interactions.
- **Long Escalations (Turn count $\ge 6$, up to 22 turns):** 30 samples (15.0%) — Severe, prolonged customer service escalations.

### 4. Temporal Representation
Conversations span key operational periods across the dataset:
- Early and mid-2017 baseline operations.
- October 2017 Great Indian Festival / Fall shopping period.
- November 2017 peak Black Friday / Cyber Monday surge.
- December 2017 holiday delivery rush.

---

## 3. Ground-Truth Labeling & Operational Triage Schema

Every example includes five verified ground-truth attributes:

| Attribute | Type | Description | Distribution |
| :--- | :--- | :--- | :--- |
| `intent` | Categorical | Finalized 10-intent taxonomy code. | 10 classes (15 to 28 samples each). |
| `difficulty` | Categorical | Complexity level (`easy`, `medium`, `hard`). | 90 easy, 70 medium, 40 hard. |
| `expected_action` | Categorical | Standardized business resolution action (11 codes). | 11 distinct operational actions. |
| `escalation_recommendation` | Binary | Operational automation decision (`auto_handle` vs `escalate_human_agent`). | 133 `auto_handle` (66.5%), 67 `escalate_human_agent` (33.5%). |
| `labeler_notes` | Free-text | Justification for intent choice, boundary precedence, and difficulty rating. | 100% populated across all 200 records. |

### Expected Resolution Actions Breakdown:
1. `TRACK_AND_DISBURSE_REFUND` (26 samples): Process refund timeline or initiate A-to-Z claim review.
2. `REPLACE_DAMAGED_DEFECTIVE_ITEM` (25 samples): Issue free replacement or return authorization for defective product.
3. `TRACK_SHIPMENT_AND_CARRIER_UPDATE` (24 samples): Provide real-time carrier tracking link and delivery window.
4. `PROCESS_ORDER_CANCELLATION` (20 samples): Cancel active order in fulfillment system or reject upon delivery.
5. `TROUBLESHOOT_DIGITAL_OR_TECHNICAL` (19 samples): Provide app cache clear/reinstall or bug report.
6. `MANAGE_PRIME_SUBSCRIPTION` (18 samples): Cancel Prime, disable auto-renew, or refund unused subscription fee.
7. `RESOLVE_PAYMENT_BILLING_ERROR` (18 samples): Investigate duplicate debit, credit Amazon Pay, or redeem gift card.
8. `ESCALATE_SERVICE_GRIEVANCE` (16 samples): File driver/agent misconduct report and route to executive relations.
9. `RECOVER_ACCOUNT_SECURITY` (15 samples): Guide password reset, 2FA verification, or route to Account Specialists.
10. `INITIATE_RETURN_PICKUP_LABEL` (15 samples): Direct to Online Returns Center for shipping label and pickup.
11. `INVESTIGATE_FALSE_DELIVERY` (4 samples): Check secure locations and initiate carrier trace investigation.

---

## 4. Automated Validation Suite Results

The automated validation suite (`src/validate_golden_set.py`) executed 10 quality gates against `data/processed/golden_evaluation_set.jsonl` and `data/processed/golden_evaluation_set.csv`:

```
================================================================================
RUNNING AUTOMATED VALIDATION SUITE: GOLDEN EVALUATION SET
================================================================================
[PASS] Verified existence of data/processed/golden_evaluation_set.jsonl
[PASS] Verified existence of data/processed/golden_evaluation_set.csv
[PASS] Verified existence of data/processed/amazonhelp_conversations_clean.csv

Total records loaded: 200
[PASS] Sample count check: 200 examples (Target: 200).
[PASS] Zero missing/null/empty values check across all 11 attributes.
[PASS] Strict intent taxonomy check: 100% of samples match approved 10 intents.
[PASS] Class distribution check: All 10 intents represented (>= 12 samples each).
[PASS] Duplicate check: 0 duplicate IDs, 0 duplicate messages across all 200 samples.
[PASS] Escalation recommendation check: {'auto_handle': 133, 'escalate_human_agent': 67}
[PASS] Difficulty tier check: {'easy': 90, 'medium': 70, 'hard': 40}
[PASS] Expected action check: 11 distinct actions verified.
[PASS] Reference integrity check: All 200 IDs verified in clean dataset.
[PASS] Format parity check: JSONL and CSV match 1-to-1 perfectly.
================================================================================
ALL 10 VALIDATION GATES PASSED SUCCESSFULLY (EXIT CODE 0).
```

---

## 5. Artifacts & Deliverables

1. **`data/processed/golden_evaluation_set.jsonl`:** 200 rich, fully annotated golden evaluation records in JSONL format.
2. **`data/processed/golden_evaluation_set.csv`:** Tabular representation of the 200 golden evaluation records for fast inspection and spreadsheet review.
3. **`LABELING_GUIDE.md` (Workspace Root):** Human-readable labeling specification detailing schema definitions, difficulty rubric, expected action taxonomy, operational triage guidelines, and precedence rules.
4. **`src/create_golden_set.py`:** Reproducible sampling and dataset curation script.
5. **`src/validate_golden_set.py`:** Programmatic validation test suite certifying dataset integrity.
6. **`reports/milestone_5_golden_evaluation_set.md`:** This milestone completion report.
