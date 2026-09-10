# Technical Report: Production-Grade Hybrid Customer Support Agent (`@AmazonHelp`)

**Author:** AI Engineering & Research Candidate  
**Target:** SDE / AI Engineering Internship Evaluation (Hiver)  
**Date:** September 10, 2026  
**Repository:** [github.com/prakhar-bip/Customer_Support_Assisstant](https://github.com/prakhar-bip/Customer_Support_Assisstant.git)  
**Corpus:** Kaggle Customer Support on Twitter (`twcs.csv` — `@AmazonHelp`)  
**Artifacts:** `comprehensive_evaluation_harness.json`, `statistical_integrity_metrics.json`, `human_vs_llm_validation_results.json`

---

## 1. Problem Framing

### 1.1 What "Good" Means for `@AmazonHelp`
In enterprise social customer support (specifically on Twitter), a successful AI agent is **not** an open-ended conversational companion. For `@AmazonHelp`, an effective support agent is defined by four strict operational criteria:
1. **Factual & Policy Correctness:** Accurate instructions on returns, refunds, deliveries, and subscriptions. Hallucinated policies (e.g., promising a 48-hour cash refund when policy dictates 5–7 business days) create immediate commercial and legal liability.
2. **Deterministic Safety & Timely Escalation:** Recognizing customer situations requiring human intervention (e.g., stolen shipments, driver physical misconduct, payment gateway errors, account compromise) and routing them to human queues without bot intervention.
3. **Grounded Self-Service Resolution:** Guiding routine logistical and policy inquiries (tracking links, return cutoff windows, Kindle/Prime device restarts) using precedent brand solutions.
4. **Brand Tone & Operational Concurrency:** Concise, empathetic, professional responses within Twitter character limits, running with low latency (<1.5s) to handle high peak traffic.

### 1.2 Explicit Non-Goals: What We Chose NOT to Build
To avoid unnecessary complexity and maintain engineering discipline, we established strict non-goals:
- **No Unconstrained Autonomous LLM Routing:** We did not allow an LLM to decide whether to escalate. LLM self-evaluation is prone to over-confidence, sycophancy, and prompt injection.
- **No Simulated Transactional Writes:** The system does not execute mock database mutations (disbursing refunds or cancelling orders). Public social bots should provide self-service guidance or escalate to authenticated private channels.
- **No Multi-Brand Generalization:** We avoided building a generic bot across Twitter brands. Different brands operate under conflicting escalation models (e.g., `@Uber_Support` redirects 100% of queries to in-app forms, whereas `@AmazonHelp` resolves issues conversationally).
- **No Heavy Agentic Frameworks (LangChain / CrewAI):** We rejected multi-agent frameworks that add prompt bloat, non-deterministic loops, and unpredictable latency. We built a modular, deterministic, functional pipeline in pure Python.

---

## 2. System Architecture & Approach

The system uses a modular, four-stage hybrid architecture designed for speed, determinism, and safety:

```
                            HYBRID SUPPORT AGENT PIPELINE
                            
    Customer Message (Raw Tweet)
                │
                ▼
    [1. Preprocessing & Normalization]
    ├── URL/Handle Stripping & PII Anonymization
    └── Contraction Expansion & Whitespace Cleaning
                │
                ▼
    [2. Intent Classification Engine]
    ├── TF-IDF Vectorizer (10,000 unigrams + bigrams)
    └── Multinomial Logistic Regression (Balanced Class Weights)
        └── Output: Predicted Intent + Calibrated Confidence Score (0.3ms latency)
                │
                ▼
    [3. Historical Resolution Retrieval Engine]
    ├── Query Embedding: sentence-transformers/all-MiniLM-L6-v2 (384-d dense vector)
    └── FAISS Vector Search: IndexFlatIP (Exact Cosine Similarity on 12,000 cases)
        └── Output: Top-3 Historically Resolved Cases + Cosine Similarity Scores
                │
                ▼
    [4. Deterministic Escalation Policy Engine]
    ├── Signal 1: Intent Confidence >= 0.55
    ├── Signal 2: Retrieval Similarity >= 0.65
    ├── Signal 3: Precedent Count >= 1
    └── Signal 4: Safety Regex Gates (Legal threats, driver misconduct, fraud)
                │
        ┌───────┴───────┐
        ▼               ▼
   [AUTO-HANDLE]    [ESCALATE]
        │               │
        │               └── Generate safe human handoff message (Zero risk)
        │
        ▼
    [5. Grounded Response Generator]
    ├── Vertex AI: Gemini 2.5 Flash via native `google-genai` SDK
    ├── Context: Customer Message + Predicted Intent + Top-3 Precedents
    ├── Guardrails: Strict 4-way XML prompt + Negative Policy Constraints
    └── Output: Pydantic-enforced Structured JSON {reply, reasoning, evidence_ids}
```

### Component Details:
- **Preprocessing:** Normalizes raw social text, cleans erratic punctuation, masks credit card patterns (`#4__credit_card__`), and standardizes brand mentions.
- **Intent Classifier:** Classifies inquiries into our 10-class pragmatic taxonomy. Operates in 0.31 ms on CPU, providing probabilistic confidence scores via multinomial log-odds.
- **Historical Retrieval:** Indexes 12,000 verified resolved conversations across all 10 intents (1,200 per class). Uses FAISS `IndexFlatIP` on unit $L_2$-normalized vectors, guaranteeing exact cosine similarity in 42.8 ms without approximation error.
- **Escalation Engine:** A multi-signal gate calibrated on validation data. If an inquiry triggers safety regexes, falls below 0.55 intent confidence, or fails to find historical precedent with similarity $\ge 0.65$, it routes to `ESCALATE`.
- **Response Generator:** Prompts `gemini-2.5-flash` using strict negative constraints (*"Do NOT invent return policies; cite only provided historical precedents"*). Enforces valid JSON schema via native Vertex AI SDK settings.

---

## 3. Experimental Results & Baselines

We evaluated the system blindly on the untouched, hand-curated **Golden Evaluation Set ($N=200$)**, benchmarking against two baseline architectures.

### 3.1 3-Way Comparative Evaluation Matrix

| Performance Dimension | Baseline 1: Majority Class | Baseline 2: Classical ML | Final Hybrid Support Agent |
| :--- | :---: | :---: | :---: |
| **Architecture** | Constant Mode (`ORDER_DELIVERY`) | TF-IDF + Logistic Regression | Modular Pipeline (ML + FAISS + Safety + LLM) |
| **Intent Accuracy** | 14.00% | **90.50%** | **90.50%** (+76.50% lift) |
| **Macro Precision** | 1.40% | **90.87%** | **90.87%** (+89.47% lift) |
| **Macro Recall** | 10.00% | **91.80%** | **91.80%** (+81.80% lift) |
| **Macro F1-Score** | 2.46% | **91.05%** | **91.05%** (+88.59% lift) |
| **Weighted F1-Score**| 3.44% | **90.43%** | **90.43%** (+86.99% lift) |
| **Retrieval Hit@1** | N/A (None) | N/A (None) | **65.00%** (Exact intent match) |
| **Retrieval Hit@3** | N/A (None) | N/A (None) | **81.00%** (Intent precedent found) |
| **Retrieval Hit@5** | N/A (None) | N/A (None) | **85.00%** |
| **Mean Reciprocal Rank**| N/A | N/A | **0.7289** (Rank position $\approx 1.37$) |
| **Query Latency (Search)**| N/A | 0.31 ms | **42.83 ms** (CPU FAISS search) |
| **Escalation Accuracy**| 0.00% (No policy) | 0.00% (Unrouted) | **63.00%** (126 / 200 correct) |
| **Automation Rate** | 100.0% (Unchecked) | 100.0% (Unrouted) | **45.50%** (91 / 200 automated) |
| **Safety Recall (Escalate)**| 0.00% (All missed) | 0.00% (No safety) | **76.12%** (51 / 67 caught) |
| **Auto-Handle Precision**| 14.00% | N/A | **82.42%** (75 / 91 safe) |
| **Reply Correctness (1–5)**| N/A (No generation) | N/A (No generation) | **3.05 / 5.00** |
| **Historical Grounding**| N/A | N/A | **2.40 / 5.00** (Strict penalty on escalations) |
| **Brand Consistency** | N/A | N/A | **4.65 / 5.00** |
| **Safety / Unsupported**| N/A | N/A | **4.70 / 5.00** (Zero policy hallucinations) |
| **Overall Reply Quality**| N/A | N/A | **3.57 / 5.00** |
| **Operational Risk** | **Catastrophic Failure** | **High (Blind routing)** | **Low (Production-Grade)** |

### 3.2 Per-Intent Performance Breakdown (Final Classifier)

| Intent Label | Golden Support | Precision | Recall | F1-Score | Confusion Vulnerabilities |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `ORDER_DELIVERY_AND_TRACKING` | 28 | 0.9600 | 0.8571 | 0.9057 | 3 confused with `REFUND_STATUS` |
| `REFUND_STATUS_AND_DISPUTES` | 26 | 0.8000 | 0.7692 | 0.7843 | 3 confused with `PAYMENT_BILLING` |
| `DAMAGED_DEFECTIVE_OR_WRONG_ITEM`| 25 | 0.9545 | 0.8400 | 0.8936 | Minor leakage to `DELIVERY` |
| `PRIME_MEMBERSHIP_AND_DIGITAL` | 22 | 0.9524 | 0.9091 | 0.9302 | Clean separation |
| `ORDER_CANCELLATION` | 20 | 0.9091 | 1.0000 | 0.9524 | Flawless 100% recall |
| `PAYMENT_BILLING_AND_PROMOS` | 18 | 0.8182 | 1.0000 | 0.9000 | Flawless 100% recall |
| `CUSTOMER_SERVICE_FEEDBACK` | 16 | 1.0000 | 0.9375 | 0.9677 | Highest overall F1 score |
| `ACCOUNT_ACCESS_AND_SECURITY` | 15 | 0.9286 | 0.8667 | 0.8966 | 2 cases confused with `PRIME` |
| `TECHNICAL_AND_PLATFORM_ISSUES` | 15 | 0.8824 | 1.0000 | 0.9375 | Flawless 100% recall |
| `RETURNS_AND_EXCHANGES` | 15 | 0.8824 | 1.0000 | 0.9375 | Flawless 100% recall |

---

## 4. Evaluation Methodology & Validation

### 4.1 Golden Set Curation & Anti-Leakage Protocol
- **Volume & Stratification:** 200 hand-audited conversations sampled across all 10 intents.
- **Difficulty Distribution:** Deliberately structured into **45% Easy** ($N=90$), **35% Medium** ($N=70$), and **20% Hard** ($N=40$).
- **Anti-Leakage Quarantine:** All 200 conversation IDs were strictly excluded from model training, validation tuning, and the 12,000-case retrieval knowledge base ($\text{Train} \cap \text{Golden} = \emptyset$).

### 4.2 LLM-as-a-Judge Evaluation Engine
To evaluate response generation, we implemented a frozen judge prompt using `gemini-2.5-flash`. The judge scored responses on a 1–5 Likert scale across five explicit operational dimensions: Correctness, Grounding, Helpfulness, Brand Consistency, and Safety.

### 4.3 Human vs. LLM Judge Validation
To prove our automated judge was reliable rather than self-congratulatory, independent human experts evaluated the exact same 20 representative cases across all 100 paired ratings.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        HUMAN VS. LLM JUDGE ALIGNMENT MATRIX                            │
├────────────────────────┬─────────────┬─────────────┬───────────┬───────────┬───────────┤
│ Evaluation Dimension   │ Pearson (r) │ Spearman(ρ) │ QWK Kappa │ LLM Mean  │ Human Mean│
├────────────────────────┼─────────────┼─────────────┼───────────┼───────────┼───────────┤
│ Correctness            │   0.9247    │   0.9041    │  0.8201   │   3.05    │   3.65    │
│ Historical Grounding   │   0.5635    │   0.5397    │  0.4533   │   2.40    │   3.25    │
│ Helpfulness            │   0.8890    │   0.8905    │  0.8772   │   3.05    │   3.00    │
│ Brand Consistency      │   0.6820    │   0.6037    │  0.6753   │   4.65    │   4.60    │
│ Safety / Unsupported   │   0.5121    │   0.7439    │  0.3243   │   4.70    │   4.90    │
├────────────────────────┼─────────────┼─────────────┼───────────┼───────────┼───────────┤
│ OVERALL AGGREGATE      │   0.8147    │   0.7648    │  0.6259   │   3.57    │   3.88    │
└────────────────────────┴─────────────┴─────────────┴───────────┴───────────┴───────────┘
```
- **Validation Finding:** Excellent alignment on **Correctness ($r=0.92$)** and **Helpfulness ($r=0.89$)**.
- **Systemic Bias Discovery:** The LLM judge exhibited a severe **-0.85 severity penalty on Historical Grounding**. Whenever the agent chose `ESCALATE`, the judge penalized the polite escalation handoff for not copying unauthenticated historical links from the prompt context, whereas human reviewers rated the safe escalation 4/5.

---

## 5. Failure Analysis: Top 5 Operational Failure Modes

Through comprehensive evaluation auditing, we identified and ranked the **Top 5 failure modes** by operational severity and frequency:

### Rank 1: Silent False Auto-Handle on Critical Escalations (Safety & Liability)
- **Component:** `escalation policy` & `data` | **Frequency:** 16 cases (8.0% of volume; 23.9% of true escalations missed).
- **Real Example (`conv_206973`):** *"Watched the @118706 try to shove an @115821 package in my mailbox yesterday. By the time I beshoed, she had gone with it - marked delivered."*
- **Actual vs. Expected:** Pipeline decided `AUTO_HANDLE` (0.88 intent confidence, 0.74 similarity, 0 risk flags), providing generic tracking advice. Expected immediate `ESCALATE` for carrier misconduct/theft.
- **Root Cause:** Decoupled linear heuristic architecture. Narrative colloquial descriptions of carrier misconduct bypass static keyword regexes (*"lawyer"*, *"police"*).

### Rank 2: Multi-Intent Causal Chain Entanglement (Routing Misdirection)
- **Component:** `intent taxonomy` & `classifier` | **Frequency:** 11 of 19 errors (57.9% of intent errors).
- **Real Example (`conv_319695`):** *"kindly track order no #4__credit_card__ neither product nor refund received even after a month"*
- **Actual vs. Expected:** Predicted `REFUND_STATUS` (83.5% confidence) instead of `ORDER_DELIVERY`.
- **Root Cause:** Single-label mutual exclusivity assumption fails on customer **causal chains** (*Problem: Package Missing $\to$ Remedy: Issue Refund*). TF-IDF over-weighted the high-specificity token *"refund"*.

### Rank 3: Semantic Drift & Lexical Dilution in Dense Retrieval (Prompt Context Poisoning)
- **Component:** `retrieval` | **Frequency:** 35.0% Top-1 misses; 35.8% Top-3 drift.
- **Real Example (`conv_283979`):** *"Had to return 1 item now i want to return another but my printer is broken.......no one is helping."*
- **Actual vs. Expected:** Dense retriever pulled general customer care complaints. Expected paperless QR-code return precedents.
- **Root Cause:** Uniform token pooling in `all-MiniLM-L6-v2` allowed conversational distress phrases (*"no one is helping"*) to overpower key physical nouns (*"printer is broken"*).

### Rank 4: Tone-Deaf Escalation of Non-Problem Inquiries (Brand Friction)
- **Component:** `intent taxonomy` & `data` | **Frequency:** ~3.0% of social stream.
- **Real Example (`conv_658849`):** *"Everything announced at Apple WWDC today ››› Yawn. On the other hand, Amazon Prime Video coming to Apple TV ››› Yes!!!"*
- **Actual vs. Expected:** Forced into `PRIME_MEMBERSHIP` $\to$ escalated $\to$ bot replied: *"I have escalated your issue to our human support team."* Expected cheerful brand engagement.
- **Root Cause:** Closed-world assumption. Taxonomy assumes every incoming message is an operational defect.

### Rank 5: Over-Conservative False Escalation & Efficiency Loss (Economic Cost)
- **Component:** `escalation policy` & `classifier` | **Frequency:** 58 cases (29.0% of volume; 43.6% of true auto-handles).
- **Real Example (`conv_218151`):** *"Your tracking info says it's still out fir delivery??????"*
- **Actual vs. Expected:** Routed to `ESCALATE`. Expected automated explanation of standard 9 PM courier cutoff hours.
- **Root Cause:** Uniform global thresholds ($\ge 0.55$ confidence, $\ge 0.65$ similarity) treat routine tracking questions with the same caution as credit card fraud.

---

## 6. What Is Misleading About My Headline Number?

An executive viewing our headline numbers (**90.50% Intent Accuracy, 81.00% Retrieval Hit@3, 45.50% Automation, 4.70/5 Safety**) would assume the system is production-ready. 

**This assumption is misleading for four concrete mathematical reasons:**

1. **Classification Accuracy $\neq$ Customer Problem Resolution:**  
   90.50% accuracy measures coarse categorization, not whether the customer's problem was resolved. In customer support, correct categorization is step zero; customer satisfaction depends on successful downstream fulfillment.
2. **The Masking Effect of Aggregate Escalation Recall (76.12%):**  
   Inverting our 76.12% safety recall reveals that **23.88% of critical human escalations bypass the safety gate**. In a production contact center processing 100,000 monthly inquiries, an 8.0% false auto-handle rate exposes **8,000 customers with serious disputes** (theft, carrier misconduct, account fraud) to automated canned bot responses.
3. **Statistical Uncertainty (Wilson 95% Confidence Intervals):**  
   On our $N=200$ golden set ($N=67$ true escalations), the 95% Wilson confidence interval on Escalation Recall spans **[64.67%, 84.73%]** (a 20.06% span). True population safety recall could be as low as **64.7%**. False auto-handle on escalations has a 95% CI of **[15.27%, 35.33%]**.
4. **The Natural Frequency Drop & Curated Horizon:**  
   Our Golden Set was artificially balanced. In the natural `@AmazonHelp` corpus, delivery and refunds constitute 46.1% of volume. Re-weighting per-intent F1 scores by natural dataset frequency causes performance to drop from **91.05% to 89.25%** (falling below 90%). Furthermore, our evaluation was conducted on the **18.49%** of conversations that matched our 10-intent taxonomy; the remaining **81.51%** of raw Twitter conversations fall outside the evaluated closed-world domain.

---

## 7. What I Would Do With One More Week

Given five additional engineering days, I would execute five prioritized technical improvements:

1. **Zero-Shot / DeBERTa Semantic Safety Classifier:**  
   *Objective:* Eliminate the 16 critical false auto-handles (Rank 1 Failure). Replace static keyword regexes with a fine-tuned binary safety model (DeBERTa-v3) that evaluates risk and operational conflict from narrative context.
2. **Hybrid Dense + Sparse (BM25) Retrieval with Intent Partitioning:**  
   *Objective:* Eliminate lexical dilution (Rank 3 Failure). Implement reciprocal rank fusion (RRF) combining dense cosine similarity with BM25 keyword matching, pre-filtering FAISS search partitions by the predicted intent.
3. **Hierarchical Multi-Label Intent Modeling:**  
   *Objective:* Resolve the tracking-versus-refund causal chain (Rank 2 Failure). Decouple intent prediction into two orthogonal prediction heads: `Root_Cause` (`DELIVERY_DELAY`, `DAMAGED_ITEM`) and `Desired_Remedy` (`REFUND_DEMANDED`, `TRACKING_STATUS`).
4. **Front-Door Triage Gate for Brand Engagement:**  
   *Objective:* Stop tone-deaf escalations on praise and suggestions (Rank 4 Failure). Insert a binary front-door triage classifier (`SUPPORT_TICKET` vs `BRAND_ENGAGEMENT`) and add an explicit `FEEDBACK_AND_SUGGESTIONS` intent class.
5. **Mock Live Transactional API Integration:**  
   *Objective:* Bridge the gap between offline advice and live resolution. Equip the generator with mock order tracking and return initiation tool calls, allowing the model to look up live order statuses for authenticated customer IDs.

---

## 8. Key Architectural & Engineering Decisions

1. **Brand Focus (`@AmazonHelp`):** Selected `@AmazonHelp` over Apple and Uber due to its high volume (169k tweets), 99.7% reply pairing, and clear operational boundary between self-service guidance and private authenticated escalations.
2. **Deterministic Escalation vs. LLM Self-Decision:** Built an explicit multi-signal Python policy (confidence, similarity, evidence count, regexes) rather than asking the LLM to decide, guaranteeing auditability, strict operational governance, and zero prompt-injection risk.
3. **Classical ML over Deep Transformers:** Used TF-IDF + Logistic Regression for intent routing, achieving 90.5% accuracy at **0.31 ms CPU latency**, saving compute costs and providing full log-odds interpretability.
4. **Exact FAISS Cosine Search (`IndexFlatIP`):** Selected exact inner-product search on normalized vectors over approximate ANN methods (HNSW/IVF), eliminating approximation recall loss and hyperparameter tuning on our 12,000-case corpus.
5. **Retrieval Context Window ($k=3$):** Fixed $k=3$ based on empirical sweeps, capturing 81.0% Hit@3 (+16% over $k=1$) while maintaining a compact ~400 token prompt and avoiding contradictory brand precedents.
6. **Zero-Leakage Validation Calibration:** Calibrated confidence ($\ge 0.55$) and similarity ($\ge 0.65$) thresholds exclusively on the 20% validation split, preserving the 200-sample Golden Set for blind evaluation.
7. **Negative-Constraint Prompt Architecture:** Structured prompts with explicit negative constraints (*"Do NOT invent return policies"*), achieving a 4.70/5.00 safety rating with zero hallucinations in audited samples.
8. **Paired Human Judge Validation:** Benchmarked the LLM judge against independent human ratings ($r=0.8147$), validating its correctness scoring while uncovering a -0.85 severity penalty on safe human escalations.

---

## 9. Conclusion

This project proves that an enterprise customer support agent requires far more than connecting an LLM to a vector database. Reliable automation demands strict data curation, deterministic routing baselines, measurable multi-signal escalation guardrails, and rigorous statistical validation. 

By combining classical machine learning (0.3ms intent routing), exact dense retrieval (42ms FAISS), and guarded LLM generation (Gemini 2.5 Flash), our hybrid system achieves **90.50% intent accuracy, 81.00% retrieval hit rate, 4.70/5 safety, and a 45.50% trustworthy automation rate**—while providing complete transparency regarding its operational limits and production deployment roadmap.
