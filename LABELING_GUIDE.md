# Golden Evaluation Set Labeling Guide & Annotation Manual

**Document Version:** 1.0.0  
**Target Brand:** `@AmazonHelp`  
**Dataset Reference:** `data/processed/golden_evaluation_set.jsonl` / `data/processed/golden_evaluation_set.csv`  
**Target Sample Size:** 200 Hand-Reviewed Benchmark Examples  
**Status:** Canonical Annotation Specification for Human Labelers & Evaluators  

---

## 1. Objective & Purpose

The **Golden Evaluation Set** is a rigorously curated, hand-reviewed benchmark of 200 customer support conversations sampled from `@AmazonHelp`. Its purpose is to serve as the definitive, uncorrupted ground truth for evaluating downstream intent classification and response generation models.

To prevent evaluation bias and benchmark saturation:
- **Zero Automated Pseudo-Labeling:** Every conversation in the golden set has been manually inspected, categorized, and annotated with operational ground truth.
- **Do Not Train on the Golden Set:** The golden set must **never** be used for model training, prompt few-shot tuning, or hyperparameter selection. It is strictly reserved for blind evaluation.

---

## 2. Ground-Truth Data Schema

Every record in `golden_evaluation_set.jsonl` and `golden_evaluation_set.csv` conforms to the following schema:

| Field Name | Type | Allowed Values / Format | Description |
| :--- | :--- | :--- | :--- |
| `conversation_id` | `string` | `conv_<root_id>` (e.g., `conv_1572692`) | Canonical identifier linking directly to the clean conversation thread. |
| `timestamp` | `string` | ISO-8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`) | Creation timestamp of the initiating message. |
| `customer_message` | `string` | Cleaned text | Initiating customer query with leading `@mentions` stripped. |
| `intent` | `string` | One of the 10 finalized taxonomy codes | Ground-truth customer support problem class. |
| `difficulty` | `string` | `easy`, `medium`, `hard` | Degree of ambiguity or complexity in identifying the intent. |
| `turn_count` | `integer` | $\ge 2$ | Total message count across customer and brand in the interaction. |
| `expected_action` | `string` | Standardized Action Code (see §4) | The concrete operational action required to resolve the issue. |
| `escalation_recommendation` | `string` | `auto_handle`, `escalate_human_agent` | Operational triage recommendation for an autonomous AI agent. |
| `escalation_reason` | `string` | Free-text explanation | Business rationale justifying automation vs. human handoff. |
| `historical_brand_response`| `string` | Text | The actual initial reply sent by `@AmazonHelp` for contextual verification. |
| `labeler_notes` | `string` | Free-text explanation | Detailed annotation reasoning, especially for boundary or difficult cases. |

---

## 3. The 10 Finalized Intent Classes

Labelers must assign exactly **one** primary intent from the approved taxonomy:

1. **`ORDER_DELIVERY_AND_TRACKING`:** Package whereabouts, shipping status, transit delays, missed delivery dates, and carrier false delivery marks ("says delivered but not received").
2. **`REFUND_STATUS_AND_DISPUTES`:** Inquiries on missing refunds, delayed bank credits, A-to-Z buyer guarantee claims, and withheld balances after return/cancellation.
3. **`DAMAGED_DEFECTIVE_OR_WRONG_ITEM`:** Physical product damage upon arrival, defective/DOA electronics, broken seals, missing parts, or incorrect/counterfeit items shipped.
4. **`PRIME_MEMBERSHIP_AND_DIGITAL`:** Prime subscription billing, auto-renewal cancellations, Prime Video/Music streaming errors, and Prime benefit inquiries.
5. **`ORDER_CANCELLATION`:** Active order cancellation requests, accidental orders, or inability to cancel orders in-transit.
6. **`PAYMENT_BILLING_AND_PROMOS`:** Checkout payment failures, duplicate bank charges, declined cards, gift card redemption errors, vouchers, and Amazon Pay wallet balance.
7. **`CUSTOMER_SERVICE_AND_COURIER_FEEDBACK`:** Escalations regarding rude couriers, unhelpful customer service phone reps, broken commitments, or general venting of frustration without an active order request.
8. **`ACCOUNT_ACCESS_AND_SECURITY`:** Login failures, password resets, locked/suspended accounts, OTP/2FA verification issues, and phishing/compromised account alerts.
9. **`TECHNICAL_AND_PLATFORM_ISSUES`:** Amazon shopping app crashes, website 500/cart glitches, navigation errors, and Kindle/FireTV/Alexa software bugs.
10. **`RETURNS_AND_EXCHANGES`:** Procedural return requests, scheduling or delayed courier pickups, return shipping labels, and exchange inquiries without a defect claim.

---

## 4. Standardized Expected Action Taxonomy

Every conversation must be mapped to one of the following 11 operational resolution actions supported by historical support workflows:

| Action Code | Action Name | Primary Description |
| :--- | :--- | :--- |
| `TRACK_SHIPMENT_AND_CARRIER_UPDATE` | Track Shipment & Update | Provide tracking portal link, carrier tracking ID, and latest delivery window estimate. |
| `INVESTIGATE_FALSE_DELIVERY` | Investigate False Delivery | Instruct customer to check safe spots/neighbors and initiate carrier trace investigation if missing after 36h. |
| `REPLACE_DAMAGED_DEFECTIVE_ITEM` | Replace Damaged/Defective Item | Issue immediate free replacement or arrange return for damaged/defective merchandise. |
| `INITIATE_RETURN_PICKUP_LABEL` | Initiate Return & Issue Label | Direct customer to Online Returns Center to generate prepaid shipping label and schedule courier pickup. |
| `TRACK_AND_DISBURSE_REFUND` | Track & Disburse Refund | Check return receipt confirmation and initiate refund credit (3–5 business days to payment method). |
| `PROCESS_ORDER_CANCELLATION` | Cancel Active Order | Process immediate order cancellation in fulfillment system or instruct customer to reject delivery. |
| `RESOLVE_PAYMENT_BILLING_ERROR` | Resolve Billing/Payment Error | Verify transaction authorization, resolve duplicate hold, or credit Amazon Pay / gift card balance. |
| `MANAGE_PRIME_SUBSCRIPTION` | Manage Prime Subscription | Assist with Prime cancellation, disable auto-renew, or process refund for unused Prime fee. |
| `RECOVER_ACCOUNT_SECURITY` | Recover Account & Security | Guide customer through password reset, 2FA recovery, or route to Account Specialists for lockout review. |
| `TROUBLESHOOT_DIGITAL_OR_TECHNICAL` | Troubleshoot App/Technical | Provide cache clearing / app re-installation steps, or log software bug ticket with engineering. |
| `ESCALATE_SERVICE_GRIEVANCE` | Escalate Service Grievance | Acknowledge service breakdown, file formal driver/agent misconduct report, and route to executive relations. |

---

## 5. Operational Triage: `auto_handle` vs. `escalate_human_agent`

A core requirement of modern customer support automation is knowing **when an autonomous AI agent can safely resolve the inquiry** versus **when the case must be escalated to a human agent**.

### `auto_handle` Criteria (Can Be Safely Automated)
An interaction should be labeled `auto_handle` if:
1. **Public Information / Guidance:** The customer requires standard policy clarification, return window rules, or instructions on how to use a feature.
2. **Self-Service Links:** The solution involves directing the customer to a secure self-service portal (e.g., Online Returns Center `amazon.com/returns`, Tracking Portal `amazon.com/orders`, Password Reset `amazon.com/forgotpassword`, Prime Management `amazon.com/prime`).
3. **General Troubleshooting:** The response provides standardized, non-destructive troubleshooting steps (e.g., clearing app cache, reinstalling Kindle app, checking TV subtitle settings).
4. **No Sensitive PII Required:** Resolution does not require the agent to inspect confidential customer records, tax forms, or payment gateway logs.

### `escalate_human_agent` Criteria (Must Be Handed Off)
An interaction must be labeled `escalate_human_agent` if:
1. **Account Lockouts & Security Threats:** Account is suspended, placed on hold, hacked, or subject to fraudulent card deductions.
2. **Account-Specific Financial Intervention:** Resolving duplicate charges, releasing withheld balances, or issuing manual goodwill concessions.
3. **Formal Disputes & Claims:** Filing or reviewing Amazon A-to-Z Buyer Guarantee claims against third-party marketplace sellers.
4. **Severe Delivery Driver Misconduct:** Delivery driver engaged in abusive behavior, property damage (threw package on roof/lawn), or falsified delivery logs.
5. **High Negative Sentiment / Legal Escalation:** Customer is extremely distressed, threatening legal/consumer court action, or experienced multiple failed support attempts.
6. **Private Direct Messaging Required:** The brand agent historically asked the customer to transition to secure Direct Messages (`DM`) or a private verification link (`https://t.co/...`) because account information is needed.

---

## 6. Difficulty Rating Rubric

To ensure the golden evaluation set rigorously tests models across the entire difficulty spectrum, every example is assigned a difficulty tier:

### 1. `easy` (~45% of evaluation set)
- **Characteristics:** Unambiguous, clear lexical and syntactic markers, prototypical phrasing.
- **Example:** *"where is my order tracking number 12345, when will it arrive?"* $\to$ `ORDER_DELIVERY_AND_TRACKING`.
- **Reasoning:** Zero ambiguity; single domain; direct match with intent definition.

### 2. `medium` (~35% of evaluation set)
- **Characteristics:** Multi-sentence, conversational nuance, informal grammar/slang, moderate customer frustration, or contextual references.
- **Example:** *"5 chats, 1 call, 1 month later, issue still stands unresolved. But was promised an ETA of 1-2 days everytime."* $\to$ `ORDER_DELIVERY_AND_TRACKING`.
- **Reasoning:** Expresses frustration and mentions customer service, but the core unfulfilled operational need is an overdue delivery ETA.

### 3. `hard` (~20% of evaluation set)
- **Characteristics:** High-overlap boundary cases, mixed intent signals, sarcasm/rhetorical questions, or multiple competing problem statements requiring strict precedence hierarchy application.
- **Example:** *"I want to replace an item I received damaged, but it will only let me return it, rather than replace. Why is this?"* $\to$ `DAMAGED_DEFECTIVE_OR_WRONG_ITEM`.
- **Reasoning:** Touches replacement, return, and damage. Precedence rule dictates that the root cause (physical item damage) takes precedence over reverse-logistics procedures.
- **Example:** *"Recharged with Amazon Pay Balance, how do I get to know that offer to win iPhone8 applied or not?"* $\to$ `PAYMENT_BILLING_AND_PROMOS`.
- **Reasoning:** Involves Amazon Pay balance and promotional contest eligibility.

---

## 7. Boundary Precedence Hierarchy

When multiple intent indicators appear in a single message, apply this strict precedence order:

1. **Root Physical Condition:** `DAMAGED_DEFECTIVE_OR_WRONG_ITEM` (physical product state broken, fake, or wrong).
2. **Active Order Lifecycle Operation:** `ORDER_CANCELLATION` or `RETURNS_AND_EXCHANGES` (actively initiating cancellation or return pickup).
3. **Financial Reimbursement:** `REFUND_STATUS_AND_DISPUTES` or `PAYMENT_BILLING_AND_PROMOS` (settlement after return/cancel or checkout failure).
4. **Logistics & Transit:** `ORDER_DELIVERY_AND_TRACKING` (whereabouts, delays, carrier marked delivered).
5. **System & Security:** `ACCOUNT_ACCESS_AND_SECURITY` or `TECHNICAL_AND_PLATFORM_ISSUES` (login, OTP, app crashes).
6. **Sentiment Venting:** `CUSTOMER_SERVICE_AND_COURIER_FEEDBACK` (used only when no actionable order request is present).

---

## 8. Quality Assurance Checklist

Before finalizing any annotation, the labeler must verify:
- [ ] Does the `intent` strictly match one of the 10 approved codes?
- [ ] Is the `expected_action` chosen from the 11 standardized action codes?
- [ ] Is the `escalation_recommendation` accurately justified by `escalation_reason`?
- [ ] Does the `difficulty` tier accurately reflect lexical and syntactic ambiguity?
- [ ] Are `labeler_notes` provided for all `hard` and `medium` examples explaining the boundary resolution?
- [ ] Does the `conversation_id` match an authentic record in the clean dataset?
