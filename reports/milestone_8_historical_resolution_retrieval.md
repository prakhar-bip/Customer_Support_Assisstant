# Milestone 8: Historical Resolution Retrieval Report (`AmazonHelp`)

**Generated On:** 2026-09-10 15:32:00  
**Target Brand:** `@AmazonHelp`  
**Embedding Architecture:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)  
**Vector Index:** FAISS `IndexFlatIP(384)` (Exact Cosine Similarity via $L_2$-normalized Inner Product)  
**Knowledge Base Corpus:** 12,000 Verified Resolved Conversations (1,200 per Intent Class)  
**Anti-Leakage Quarantine:** 200 Quarantined Golden Evaluation Samples Excluded ($\text{KB} \cap \text{Golden} = \emptyset$)  
**Evaluation Set:** `data/processed/golden_evaluation_set.jsonl` ($N = 200$ Hand-Reviewed Cases)  
**FAISS Index Artifact:** `models/historical_knowledge_base.faiss` (17.58 MB)  
**Metadata Store Artifact:** `data/processed/historical_knowledge_base.jsonl` (13.03 MB)  
**Standalone Module:** `src/retrieve_resolutions.py`  
**Evaluation Script:** `src/evaluate_retrieval.py`  
**Evaluation Metrics:** `data/analysis/retrieval_evaluation_metrics.json`  

---

## 1. Executive Summary

A primary requirement of the Hiver assignment is that **automated customer support responses must be strictly grounded in how the selected brand historically resolved similar customer inquiries**.

In Milestone 8, we engineered and validated an end-to-end historical resolution retrieval engine using `sentence-transformers` and `FAISS`:
- **Curated Knowledge Base:** Formed a knowledge base of **12,000 high-quality resolved interactions** from `@AmazonHelp`, capturing for every case: `conversation_id`, `customer_problem`, `relevant_conversation_context`, `brand_response`, and `intent`.
- **Exact Vector Index:** Generated 384-dimensional dense vectors using `all-MiniLM-L6-v2` with unit $L_2$ normalization, indexed via FAISS `IndexFlatIP` to guarantee exact, mathematically deterministic cosine similarity.
- **Blind Benchmark Evaluation ($N = 200$):** Evaluated strictly on the quarantined Golden Evaluation Set:
  - **Top-1 Intent Hit Rate (Precision@1):** **65.00%**
  - **Top-3 Intent Hit Rate (Hit@3):** **81.00%**
  - **Top-5 Intent Hit Rate (Hit@5):** **85.00%**
  - **Mean Reciprocal Rank (MRR):** **0.7289**
  - **Mean Average Precision (MAP@5):** **0.7053**
  - **Average Query Latency:** **42.83 ms** (including embedding encoding on CPU).
- **Scope Discipline:** Strictly zero response generation was performed; all efforts centered purely on deterministic, reliable historical-case retrieval.

---

## 2. System Architecture & Methodology

```
[New Customer Inquiry]
          │
          ▼
[sentence-transformers/all-MiniLM-L6-v2]
  - 384-dimensional dense semantic embedding
  - Unit L2 Normalization: ||v|| = 1.0
          │
          ▼
[FAISS IndexFlatIP(384)]
  - Exact inner-product search (u · v = cos θ)
  - 12,000 indexed historical cases
  - 100% Deterministic (no clustering or graph approximations)
          │
          ▼
[Top-k Candidate Extraction & Hydration]
  - Cosine Similarity Score ∈ [-1.0, 1.0]
  - Source Conversation ID (conv_xxxxxx)
  - Customer Problem Statement
  - Relevant Multi-Turn Conversation Context
  - Historical Brand Response & Resolution Status
  - Inferred / Grounded Intent Code
```

### Knowledge Base Curation Criteria:
1. **Resolution Quality Filter:** Retained only conversations where `resolution_status` was `resolved_guidance_provided` or `resolved_customer_confirmed` (66,699 clean candidates).
2. **Text Quality Pruning:** Retained queries with customer text $\ge 15$ characters and brand responses $\ge 20$ characters, filtering out greeting stubs.
3. **Intent-Stratified Balancing:** Partitioned 1,200 cases per intent across all 10 customer support categories, sorted by classifier confidence. This ensures that rare intents (e.g. `RETURNS_AND_EXCHANGES`, `TECHNICAL_AND_PLATFORM_ISSUES`) are as densely represented in vector space as high-volume intents.
4. **Anti-Leakage Quarantine:** Strictly filtered out all 200 conversation IDs present in `golden_evaluation_set.jsonl` ($\text{Overlap} = 0$).

---

## 3. Quantitative Evaluation on Golden Evaluation Set ($N = 200$)

The retrieval engine was evaluated blindly against the 200 hand-reviewed benchmark queries:

### Global Retrieval Performance

| Metric | Score | Operational Significance |
| :--- | :---: | :--- |
| **Top-1 Intent Hit Rate (Hit@1)** | **65.00%** | In 65% of queries, the #1 nearest historical neighbor shares the identical intent. |
| **Top-3 Intent Hit Rate (Hit@3)** | **81.00%** | In 81% of queries, at least one of the top-3 retrieved cases matches the intent. |
| **Top-5 Intent Hit Rate (Hit@5)** | **85.00%** | In 85% of queries, at least one of the top-5 retrieved cases matches the intent. |
| **Mean Reciprocal Rank (MRR)** | **0.7289** | On average, the first intent-congruent historical precedent appears at rank $\approx 1.37$. |
| **Mean Average Precision (MAP@5)** | **0.7053** | High density of relevant historical cases across the retrieved top-5 candidate set. |
| **Mean Top-1 Cosine Similarity** | **0.7209** | Robust semantic proximity across all queries. |
| **Mean Top-5 Cosine Similarity** | **0.6858** | Smooth semantic gradient without sharp drop-offs. |
| **Average Query Latency** | **42.83 ms** | Sub-50ms CPU latency enables real-time synchronous retrieval. |

---

### Performance by Difficulty Tier

| Difficulty Tier | Sample Count ($N$) | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR | Mean Top-1 Sim |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`easy`** | 90 | **72.22%** | **87.78%** | **88.89%** | **0.7935** | 0.7154 |
| **`medium`** | 70 | **71.43%** | **88.57%** | **92.86%** | **0.7988** | 0.7103 |
| **`hard`** | 40 | **37.50%** | **52.50%** | **62.50%** | **0.4612** | 0.7520 |

#### Key Insight on the "Hard" Tier:
The 40 "hard" cases in our golden set represent complex, multi-issue boundary disputes (e.g., `"My package is marked delivered but not here, so cancel it and refund me"`). Dense vector embeddings map the entire sentence semantics, pulling cases from delivery tracking, refunds, and cancellations simultaneously. Notice that the mean cosine similarity on `hard` queries is actually higher (0.7520) because the sentence contains rich vocabulary, but the top-1 hit rate drops because dense embedding alone does not enforce precedence rules without intent filtering.

---

### Performance by Intent Category

| Intent Code | Support ($N$) | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`PAYMENT_BILLING_AND_PROMOS`** | 18 | **94.44%** | **94.44%** | **94.44%** | **0.9444** |
| **`RETURNS_AND_EXCHANGES`** | 15 | **86.67%** | **93.33%** | **93.33%** | **0.8889** |
| **`ACCOUNT_ACCESS_AND_SECURITY`** | 15 | **80.00%** | **80.00%** | **93.33%** | **0.8333** |
| **`TECHNICAL_AND_PLATFORM_ISSUES`** | 15 | **73.33%** | **86.67%** | **86.67%** | **0.8000** |
| **`ORDER_CANCELLATION`** | 20 | **70.00%** | **90.00%** | **95.00%** | **0.7958** |
| **`PRIME_MEMBERSHIP_AND_DIGITAL`** | 22 | **68.18%** | **90.91%** | **90.91%** | **0.7879** |
| **`CUSTOMER_SERVICE_AND_COURIER_FEEDBACK`** | 16 | **56.25%** | **75.00%** | **81.25%** | **0.6583** |
| **`ORDER_DELIVERY_AND_TRACKING`** | 28 | **50.00%** | **67.86%** | **67.86%** | **0.5714** |
| **`REFUND_STATUS_AND_DISPUTES`** | 26 | **50.00%** | **76.92%** | **88.46%** | **0.6378** |
| **`DAMAGED_DEFECTIVE_OR_WRONG_ITEM`** | 25 | **48.00%** | **68.00%** | **72.00%** | **0.5833** |

---

## 4. Qualitative Case Studies

### 1. Successful Retrieval (High Precision & Direct Resolution)

#### Case 1: Alexa App Launch Query (`conv_244944`)
- **Incoming Customer Query:** `"when will be Alexa app available on apple app store india?"`
- **Ground Truth Intent:** `TECHNICAL_AND_PLATFORM_ISSUES` | **Expected Action:** `TROUBLESHOOT_DIGITAL_OR_TECHNICAL`
- **Rank 1 Retrieved Case (`conv_1018210`):**
  - **Similarity Score:** `0.9543` (Extremely high semantic congruence)
  - **Historical Customer Problem:** `"When is the alexa app going to be available in India ?"`
  - **Historical Brand Resolution:** `"@360909 The Alexa app will be available in the App Store / Play Store when the devices start shipping the week of Oct 30. ^HN"`
  - **Dialogue Context:** Turn 1 (Customer inquiry) $\to$ Turn 2 (Official release timeline provided).
- **Assessment:** **Exemplary retrieval.** The retriever pulled an identical real-world query with the precise corporate launch date and official store availability instructions.

#### Case 2: Tracking Delay Query (`conv_1270275`)
- **Incoming Customer Query:** `"My package is 3 days late, tracking has not updated at all"`
- **Ground Truth Intent:** `ORDER_DELIVERY_AND_TRACKING` | **Expected Action:** `TRACK_CARRIER_OR_ESCALATE_DELAY`
- **Rank 1 Retrieved Case (`conv_2945566`):**
  - **Similarity Score:** `0.8178`
  - **Historical Customer Problem:** `"My package has been delayed two days in a row, and there's no tracking update yet today despite a CSR telling me I'd definitely get it today. Just tell me they lost it and send me another one already."`
  - **Historical Brand Resolution:** `"@223600 Hi, apologies for the delay. Is the order fulfilled by Amazon: https://t.co/Y5jpI9gRhE? ^JJ"`
- **Assessment:** **Highly relevant precedent.** Provides the official Amazon fulfillment triage link and appropriate diagnostic question for late carrier tracking.

---

### 2. Partially Useful Retrieval (Related Domain / Requires Adaptation)

#### Case 3: Compound Account Recovery & Lost Gift Card Balance (`conv_219492`)
- **Incoming Customer Query:** `"hi there just got my account back from an issue. now i have lost my $5.00 gift card as well as my preorder for a game on sale.."`
- **Ground Truth Intent:** `PAYMENT_BILLING_AND_PROMOS` | **Expected Action:** `RESOLVE_PAYMENT_BILLING_ERROR`
- **Rank 1 Retrieved Case (`conv_2279417`):**
  - **Similarity Score:** `0.6503`
  - **Historical Customer Problem:** `"And I have received the gift card into my account finally. Thanks for the prompt action. Appreciate that! 🤩"`
  - **Historical Brand Resolution:** `"@662708 If the details have been shared using the link, we'll work on it and get back with a correspondence to the registered email address soon. ^AR"`
- **Rank 2 Retrieved Case (`conv_2766433`):**
  - **Similarity Score:** `0.6243`
  - **Historical Customer Problem:** `"My amazon.fr account was cancelled by amazon. But I bought an amazon gift card last day. I didn't receive the gift card. Would you mind help me check it. Order NO.#171-8677234-0033944 @AmazonHelp"`
  - **Historical Brand Resolution:** `"@773652 Hi Molly, we don't have access to customer accounts over Twitter. Please reach out to us here so we can look into this further: https://t.co/hgIrz98cdI ^SA"`
- **Assessment:** **Partially useful.** The incoming query contains three sub-problems: account recovery, missing \$5 gift card balance, and a cancelled pre-order. The retriever successfully pulled cases involving missing gift cards on recovered/cancelled accounts and provided the authenticated customer service link (`https://t.co/hgIrz98cdI`), but the historical response does not address the game sale price match. In Milestone 9, an LLM grounded in this case would need to adapt the response to address the pre-order discount.

---

### 3. Bad / Misleading Retrieval (Semantic Drift & Sarcasm)

#### Case 4: Website Pagination Bug (`conv_179301`)
- **Incoming Customer Query:** `"there is a serious bug on your website; when trying to navigate using Next page or Page Number it redirects to Home Page."`
- **Ground Truth Intent:** `TECHNICAL_AND_PLATFORM_ISSUES` | **Expected Action:** `TROUBLESHOOT_DIGITAL_OR_TECHNICAL`
- **Rank 1 Retrieved Case (`conv_1502859`):**
  - **Similarity Score:** `0.5227` (Low similarity)
  - **Retrieved Intent:** `ACCOUNT_ACCESS_AND_SECURITY` (Mismatch)
  - **Historical Customer Problem:** `"login > says im authorized to my merchant based on CID/MCID > 'go to homepage button' isn't working, redirects to same page."`
  - **Historical Brand Resolution:** `"@128622 I'm sorry to hear this! Have you attempted clearing your cache and cookies or trying another browser? ^TH"`
- **Why It Failed:** The dense embedding focused on lexical overlap around redirection (`"redirects to Home Page"` vs `"redirects to same page"` / `"go to homepage button"`), retrieving a merchant seller authorization issue rather than general customer catalog navigation. Even though the troubleshooting step (clear cache/cookies) is coincidentally reasonable, the domain context was misaligned.

#### Case 5: Sarcastic Money-Saving Joke (`conv_227018`)
- **Incoming Customer Query:** `"Great sale started on Amazon today, if you login into the sites you can save up to 60%,But you can save 100% if you don't login 😂"`
- **Ground Truth Intent:** `ACCOUNT_ACCESS_AND_SECURITY` / Social Commentary | **Expected Action:** `RECOVER_ACCOUNT_SECURITY`
- **Rank 1 Retrieved Case (`conv_2451334`):**
  - **Similarity Score:** `0.5444`
  - **Retrieved Intent:** `PAYMENT_BILLING_AND_PROMOS`
  - **Historical Customer Problem:** `"shopped on Amazon betn dates ... said 15% Cashback if I use Amazon Pay but I got just 3.6 %"`
- **Why It Failed:** The customer posted a humorous tweet mocking shopping behavior. The dense embedder grasped the financial vocabulary (`"save up to 60%"`, `"sale"`, `"login"`) and pulled a cashback billing dispute. Dense embeddings struggle with humor and sarcasm without conversational context.

---

## 5. Standalone Reproducible Inference Engine

The retrieval module is encapsulated in `src/retrieve_resolutions.py`:

```python
from src.retrieve_resolutions import retrieve_similar_cases

results = retrieve_similar_cases(
    query_text="My package was marked delivered yesterday but I haven't received it!",
    top_k=3
)

for r in results:
    print(f"Rank {r['rank']} | Sim: {r['similarity_score']:.4f} | Conv ID: {r['conversation_id']}")
    print(f"Historical Resolution: {r['brand_response']}\n")
```

### CLI Verification:
```bash
python src/retrieve_resolutions.py --query "Where is my package? Tracking hasn't updated in 4 days" --top-k 3
```

---

## 6. Key Takeaways for Milestone 9 (Response Generation)

1. **Top-3/Top-5 Coverage is Robust (81–85% Hit Rate):** While top-1 hit rate is 65%, expanding the retrieval context to top-3 or top-5 yields an 85% intent hit rate, giving response generators strong historical context to synthesize from.
2. **Intent-Conditioned Re-Ranking:** When combined with our Milestone 7 intent classifier (91% Macro F1), passing an `intent_filter` to the retriever boosts precision on boundary cases.
3. **Guardrails for Low Similarity (<0.60):** Cases with similarity $< 0.60$ or involving sarcasm should be flagged for fallback or safe clarifying questions rather than asserting historical precedent.

---

## 7. Deliverables Summary

| Milestone | Deliverable | Path | Description |
| :--- | :--- | :--- | :--- |
| **M8** | Knowledge Base Builder | [`src/build_knowledge_base.py`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/src/build_knowledge_base.py) | Stratified KB builder and FAISS indexing pipeline. |
| **M8** | FAISS Vector Index | [`models/historical_knowledge_base.faiss`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/models/historical_knowledge_base.faiss) | Exact cosine similarity FAISS index (17.6 MB). |
| **M8** | Metadata Store | [`data/processed/historical_knowledge_base.jsonl`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/data/processed/historical_knowledge_base.jsonl) | 12,000 indexed historical cases with context and resolutions (13.0 MB). |
| **M8** | Knowledge Base Audit | [`data/analysis/knowledge_base_summary.json`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/data/analysis/knowledge_base_summary.json) | Audit metrics and distribution stats. |
| **M8** | Retrieval Engine | [`src/retrieve_resolutions.py`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/src/retrieve_resolutions.py) | Standalone Python module and CLI. |
| **M8** | Evaluation Suite | [`src/evaluate_retrieval.py`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/src/evaluate_retrieval.py) | 200-sample golden set benchmark evaluator. |
| **M8** | Evaluation Metrics | [`data/analysis/retrieval_evaluation_metrics.json`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/data/analysis/retrieval_evaluation_metrics.json) | Full machine-readable metrics, rankings, and case studies. |
| **M8** | Technical Report | [`reports/milestone_8_historical_resolution_retrieval.md`](file:///c:/Users/prakh/OneDrive/Desktop/Hiver/reports/milestone_8_historical_resolution_retrieval.md) | Comprehensive engineering report. |
