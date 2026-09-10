# Milestone 16: Architectural & Engineering Decision Log

**Document Version:** 1.0  
**Effective Date:** 2026-09-10  
**Target System:** Hybrid Support Agent (`@AmazonHelp`)  
**Scope:** Retrospective Audit of Non-Trivial Architectural, Machine Learning, and Evaluation Decisions  
**Target Audience:** Engineering Leadership, Machine Learning System Architects, Hiring Committee

---

## 1. Executive Summary

Building an enterprise-grade AI customer support system requires navigating trade-offs between accuracy, latency, interpretability, safety, and operational cost. Rather than adopting default configurations or chasing complex modeling trends, every architectural choice in the `@AmazonHelp` support pipeline was driven by empirical constraints and risk management.

This document logs **14 major, non-obvious engineering and research decisions** made across all phases of the project—from dataset reconstruction and intent modeling to dense retrieval indexing, escalation calibration, LLM prompting, and human-in-the-loop evaluation.

Trivial decisions (e.g., programming language selection or basic directory structures) have been omitted. Each entry details the core decision, the driving motivation, the viable alternatives evaluated, the concrete rationale for their rejection, and the measured empirical trade-offs.

---

## 2. Comprehensive Decision Log

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROJECT DECISION MAP                              │
├──────────────────────┬──────────────────────────────────────────────────────┤
│ Data & Formulation   │ • 1. Brand Selection (@AmazonHelp)                   │
│                      │ • 2. Conversation Filtering & Thread Reconstruction   │
│                      │ • 3. Coarse 10-Class Intent Taxonomy                 │
│                      │ • 4. Stratified Difficulty Sampling (N=200)          │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ Machine Learning     │ • 5. Classical TF-IDF + Logistic Regression Baseline │
│ & Retrieval          │ • 6. Sentence-Transformers all-MiniLM-L6-v2 Embeddings│
│                      │ • 7. FAISS IndexFlatIP Exact Cosine Vector Search     │
│                      │ • 8. Conditioning Generation on Fixed Top-k (k = 3)  │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ Policy & Generation  │ • 9. Explicit Multi-Signal Deterministic Escalation  │
│                      │ • 10. Zero-Leakage Threshold Grid Calibration        │
│                      │ • 11. Vertex AI Gemini 2.5 Flash via Native GenAI    │
│                      │ • 12. Strict Negative Constraint Prompt Architecture  │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ Evaluation & Quality │ • 13. 5-Dimensional LLM-as-a-Judge Evaluation Rubric │
│                      │ • 14. Paired Human vs. LLM Correlation Validation     │
└──────────────────────┴──────────────────────────────────────────────────────┘
```

---

### Decision 1: Brand Selection — Choosing `@AmazonHelp` over `@AppleSupport` and `@Uber_Support`

- **Decision:** Selected `@AmazonHelp` as the exclusive brand corpus for building the historical knowledge base, training the intent classifier, and evaluating the agent pipeline.
- **Why:** `@AmazonHelp` represents the single largest customer support volume in the Twitter Customer Support dataset (169,840 outbound tweets, 169,287 direct replies to customer queries, 99.7% pairing rate). Its domain spans e-commerce delivery logistics, payment disputes, returns, Prime subscriptions, digital streaming, and smart home hardware (Echo/Alexa, Fire TV). Crucially, `@AmazonHelp` exhibits a clear, observable operational separation between issues that can be auto-handled via self-service guidance (tracking URLs, return windows) versus issues requiring private authenticated escalation (refund overrides, account access).
- **Alternatives considered:**
  1. `@AppleSupport` (106,860 tweets)
  2. `@Uber_Support` (56,270 tweets)
  3. `@SpotifyCares` (43,265 tweets)
- **Why we rejected them:**
  - `@AppleSupport` interactions predominantly involve deep, hardware-specific diagnostic trees (e.g., iOS firmware restore loops, battery degradation, physical Genius Bar appointments) requiring private device serial numbers that are unavailable in public tweets.
  - `@Uber_Support` suffers from extreme response templating: nearly 100% of outbound tweets redirect customers to an in-app support link (`t.co/...`) without providing substantive in-channel problem resolution.
  - `@SpotifyCares` has high conversational quality but a narrow domain (music playback, account billing, playlist bugs), lacking the multi-faceted logistical and transactional complexity needed to stress-test an enterprise routing agent.
- **Evidence/tradeoff:** Analyzing 2.81 million tweets in Milestone 1 proved that `@AmazonHelp` yielded 79,665 clean, multi-turn conversations and the fastest median response time (11.47 minutes). The tradeoff was managing regional customer support nuances across Amazon US, UK, and India, which we normalized during preprocessing.

---

### Decision 2: Conversation Filtering — Thread Reconstruction & Strict Resolution State Tagging

- **Decision:** Reconstructed full conversational trees by following parent pointers (`in_response_to_tweet_id`) up to root customer inquiries, and filtered the knowledge base exclusively for interactions verified as `resolved_guidance_provided` or `resolved_customer_confirmed`.
- **Why:** In raw Twitter data, customer-brand interactions are fractured: customers send multiple tweets, brands send split replies, and customers post unrelated commentary. Indexing uncurated raw tweet pairs introduces toxic rants, unanswered questions, and truncated advice into the retrieval index, which would poison the LLM's prompt context with ungrounded or adversarial examples.
- **Alternatives considered:**
  1. Indexing all customer-brand tweet pairs indiscriminately without resolution filtering.
  2. Indexing only the customer's initial tweet and the brand's initial reply, ignoring multi-turn context.
  3. Filtering conversations using sentiment analysis (indexing only interactions ending in positive sentiment).
- **Why we rejected them:**
  - Indiscriminate indexing resulted in 42% of retrieved historical cases containing generic redirects (*"Please DM us your email"*) rather than actionable resolutions.
  - Single-turn truncations lost crucial context: in 31% of Amazon cases, the customer only clarified their true order problem in turn 2 or turn 3.
  - Sentiment filtering discarded legitimate, successful resolutions where the customer remained annoyed despite receiving accurate policy advice.
- **Evidence/tradeoff:** Filtering reduced the knowledge base pool from 79,665 clean conversations to 12,000 verified resolution cases (an 85% volume reduction). However, this guaranteed that every retrieved case provided actionable procedural guidance, achieving a 4.70/5.00 safety rating during response generation.

---

### Decision 3: Intent Taxonomy — 10 Coarse Pragmatic Classes vs. Granular 30+ Micro-Intents

- **Decision:** Established an explicit 10-class intent taxonomy (`ORDER_DELIVERY_AND_TRACKING`, `REFUND_STATUS_AND_DISPUTES`, `DAMAGED_DEFECTIVE_OR_WRONG_ITEM`, `PRIME_MEMBERSHIP_AND_DIGITAL`, `ORDER_CANCELLATION`, `PAYMENT_BILLING_AND_PROMOS`, `CUSTOMER_SERVICE_AND_COURIER_FEEDBACK`, `ACCOUNT_ACCESS_AND_SECURITY`, `TECHNICAL_AND_PLATFORM_ISSUES`, `RETURNS_AND_EXCHANGES`) with defined precedence ordering.
- **Why:** Enterprise customer service operations route inquiries to specialized functional tiers (Logistics, Billing, Digital Media, Account Security). A 10-class taxonomy provides a pragmatic mapping to operational departments while maintaining sufficient statistical support ($N > 500$ training cases per class) to train reliable classifiers.
- **Alternatives considered:**
  1. Fine-grained micro-taxonomy (30–50 specific intents like `TRACKING_STUCK_IN_TRANSIT`, `REFUND_TO_GIFT_CARD_DISPUTE`, `ALEXA_WIFI_DISCONNECT`).
  2. Coarse 3-class routing taxonomy (`SHIPPING`, `BILLING`, `TECHNICAL_SUPPORT`).
  3. Unsupervised zero-shot clustering without predefined intent classes.
- **Why we rejected them:**
  - 30+ micro-intents resulted in severe class sparsity in long-tail categories ($N < 50$ samples), high inter-annotator disagreement (Cohen's $\kappa < 0.55$), and classifier confusion on overlapping edge cases.
  - 3 coarse classes were too broad to be actionable: a ticket labeled `BILLING` could be a routine coupon question (auto-handle) or a credit card fraud claim (immediate human escalation).
  - Unsupervised clustering produced semantically unstable clusters that shifted across runs and could not be mapped to deterministic escalation policies.
- **Evidence/tradeoff:** The 10-intent taxonomy achieved **93.50% validation accuracy** and **90.50% golden accuracy**. The tradeoff is boundary ambiguity on causal chains (e.g., late delivery resulting in a refund demand), which accounted for 57.9% of intent errors and was subsequently addressed in our failure analysis.

---

### Decision 4: Golden Evaluation Set Design — Stratified Difficulty Sampling ($N=200$) with Zero-Leakage Quarantine

- **Decision:** Built a fixed, hand-curated Golden Evaluation Set of exactly 200 conversations stratified across all 10 intents and deliberately balanced across three difficulty tiers: **45% Easy** ($N=90$), **35% Medium** ($N=70$), and **20% Hard** ($N=40$). All 200 records were quarantined from model training and vector indexing ($\text{Train} \cap \text{Golden} = \emptyset$).
- **Why:** Evaluating solely on random test splits masks critical model failures because real distributions are dominated by simple, repetitive queries. A fixed golden set with stratified difficulty acts as a diagnostic stress test, revealing how systems degrade when confronted with complex, emotionally charged, or ambiguous complaints.
- **Alternatives considered:**
  1. Standard random 80/20 train/test split.
  2. Cross-validation across the entire dataset.
  3. LLM-generated synthetic evaluation queries.
- **Why we rejected them:**
  - Random splits over-represented dominant easy tracking queries (40% of data) and leaked conversational customer styles into test sets.
  - Cross-validation on 80,000 cases would require retraining vector indices and rerunning expensive LLM generation evaluations repeatedly without providing a stable, inspectable benchmark.
  - Synthetic test cases lacked the colloquialisms, typos, emotional intensity, and unstructured ambiguity of real Twitter interactions.
- **Evidence/tradeoff:** Stratified difficulty sampling revealed the **"Difficulty Cliff"**: retrieval Hit@3 dropped from **87.78%** on Easy cases down to **52.50%** on Hard cases. This diagnostic insight would have remained completely hidden under a naive random split.

---

### Decision 5: Intent Classifier Architecture — Classical TF-IDF + Logistic Regression with Balanced Class Weights

- **Decision:** Selected word and bi-gram TF-IDF vectorization (10,000 max features, sublinear term frequency, English stopword removal) paired with multinomial Logistic Regression (`class_weight='balanced'`, $C=1.0$) over deep neural models.
- **Why:** The classical ML pipeline delivers sub-millisecond CPU inference (**0.31 ms per query**), requires no GPU infrastructure, has zero API operating costs, and provides complete mathematical interpretability via log-odds feature weights. Setting `class_weight='balanced'` ensured that minority classes (`RETURNS_AND_EXCHANGES`, `TECHNICAL_ISSUES`) were penalized inversely proportional to their training frequency.
- **Alternatives considered:**
  1. Fine-tuned transformer models (`RoBERTa-base` / `DeBERTa-v3`).
  2. Zero-shot or few-shot LLM classification via Vertex AI API.
  3. Naive Bayes or Random Forest classifiers.
- **Why we rejected them:**
  - Fine-tuned transformers added 40–70ms of latency per query, required dedicated GPU hosting, and introduced deployment complexity for a marginal estimated gain of 2–3% accuracy.
  - Few-shot LLM classification introduced nondeterministic outputs, cost ~$0.0015 per query, had high API latency (~800ms), and risked JSON format parse failures.
  - Naive Bayes assumed conditional independence, failing on overlapping n-grams between delivery tracking and return requests.
- **Evidence/tradeoff:** The TF-IDF + Logistic Regression pipeline achieved **90.50% accuracy** and **91.05% macro F1** on the untouched golden set, outperforming the majority baseline (14.00% accuracy, 2.46% macro F1) by **+76.50%** and **+88.59%** respectively, while executing in 0.3ms on CPU.

---

### Decision 6: Embedding Model — Dense `sentence-transformers/all-MiniLM-L6-v2`

- **Decision:** Selected `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors with unit $L_2$ normalization) for embedding customer problems into vector space.
- **Why:** It offers an optimal Pareto frontier across embedding speed, memory footprint, and semantic retrieval accuracy. A 384-dimensional index for 12,000 cases occupies only 17.6 MB in memory, and query vector encoding takes under 40 milliseconds on a standard CPU without requiring GPU acceleration.
- **Alternatives considered:**
  1. Commercial embedding APIs (OpenAI `text-embedding-3-small`, Google `text-embedding-004`).
  2. Larger open-source models (`sentence-transformers/all-mpnet-base-v2`, 768-d).
  3. Sparse lexical embeddings (BM25 alone).
- **Why we rejected them:**
  - Commercial APIs introduced external network roundtrips (adding 150–300ms latency), recurring per-token billing costs, rate limits, and external service availability dependencies.
  - `all-mpnet-base-v2` doubled vector dimensionality (increasing index RAM by 100%) and tripled inference latency (120ms vs 38ms) while improving golden set Top-3 Hit Rate by less than 1.5% in initial benchmarks.
  - Pure BM25 failed completely on vocabulary mismatch (e.g., matching customer query *"where is my parcel"* with historical precedent *"package in transit"*).
- **Evidence/tradeoff:** `all-MiniLM-L6-v2` delivered an average query latency of **42.83 ms** and an **81.00% Top-3 Hit Rate**. The tradeoff was lexical dilution on conversational filler (e.g. *"no one is helping me"* overpowering physical keywords like *"broken printer"*), which was documented as Failure Mode 3.

---

### Decision 7: Vector Indexing & Search Architecture — FAISS `IndexFlatIP` with Unit Normalization

- **Decision:** Implemented vector similarity search using FAISS with an exact Inner Product index (`IndexFlatIP`) on unit $L_2$-normalized vectors, guaranteeing exact cosine similarity calculation.
- **Why:** Exact search eliminates approximation error, guarantees 100% deterministic ranking across identical queries, requires zero index hyperparameter tuning (no `nlist`, `nprobe`, or `efSearch` parameters), and executes in less than 5 milliseconds across 12,000 vectors.
- **Alternatives considered:**
  1. Inverted File Index with Voronoi cells (`IndexIVFFlat`).
  2. Hierarchical Navigable Small World graphs (`IndexHNSWFlat`).
  3. Vector database services (Pinecone, Qdrant, Weaviate, Milvus).
  4. Relational database with vector extensions (`pgvector` on PostgreSQL).
- **Why we rejected them:**
  - Approximate Nearest Neighbor (ANN) indices (`IVFFlat`, `HNSW`) trade away retrieval recall to achieve sub-linear scaling on millions of vectors. On a 12,000-case corpus, the linear scan takes only 4.2ms, making approximation loss unjustified.
  - Dedicated vector database services introduced infrastructure overhead, container management, and cloud networking latency for a dataset that comfortably fits in 18 MB of local memory.
  - `pgvector` added SQL roundtrip overhead and database dependency without providing performance advantages over in-process FAISS.
- **Evidence/tradeoff:** Achieved **0.7289 Mean Reciprocal Rank (MRR)** and sub-5ms search latency. The tradeoff is linear $O(N)$ scaling, which will require transitioning to HNSW if the knowledge base scales beyond 500,000 vectors.

---

### Decision 8: Retrieval Context Window ($k=3$) — Fixing Top-3 Precedents for Response Generation

- **Decision:** Fixed the retrieval context window at exactly $k = 3$ historical resolution precedents when populating the generation prompt.
- **Why:** An empirical sweep across $k \in \{1, 2, 3, 5, 10\}$ demonstrated that $k=3$ captures **81.00% Top-k Hit Rate** (a massive +16.0% gain over $k=1$ at 65.00%), while maintaining a concise prompt context (~400 tokens) that fits well within LLM context windows and prevents contradictory precedents.
- **Alternatives considered:**
  1. $k = 1$ (single nearest neighbor).
  2. $k = 5$ (wider context window).
  3. Dynamic similarity thresholding (retrieve all precedents with cosine similarity $> 0.70$).
- **Why we rejected them:**
  - $k=1$ resulted in a 35.0% intent miss rate, providing irrelevant prompt evidence on one-third of all queries.
  - $k=5$ added only +4.0% marginal hit rate (85.00%) while nearly doubling prompt token count, increasing LLM inference latency by 280ms, and frequently presenting contradictory brand responses (e.g., one agent offering a replacement, another offering a refund).
  - Dynamic thresholding produced variable context lengths (from 0 to 12 cases), causing prompt instability and unpredictable token usage.
- **Evidence/tradeoff:** Setting $k=3$ maintained prompt consistency and kept average generation latency under 1.5 seconds. The tradeoff was missing the 4% of relevant precedents that were ranked at positions 4 and 5.

---

### Decision 9: Explicit Multi-Signal Escalation Engine vs. LLM-Prompted Escalation Decision

- **Decision:** Implemented the escalation decision engine as an explicit, deterministic Python module combining four measurable signals:
  1. Intent classifier confidence ($\ge \tau_{\text{conf}}$)
  2. Dense retrieval cosine similarity ($\ge \tau_{\text{sim}}$)
  3. High-relevance precedent count ($\ge N_{\text{evidence}}$)
  4. Static safety keyword regexes (detecting legal threats, physical safety, carrier misconduct)
- **Why:** Delegating escalation decisions to an unconstrained LLM prompt introduces non-determinism, sycophantic bias, and susceptibility to prompt injection. An explicit policy provides full operational transparency, auditability, and immediate human-controllable thresholds.
- **Alternatives considered:**
  1. Asking the LLM to output `"decision": "AUTO_HANDLE" | "ESCALATE"` directly in its JSON response.
  2. A machine-learned binary classifier (e.g., Logistic Regression or SVM trained on human escalation labels).
  3. Pure keyword-based heuristic rules.
- **Why we rejected them:**
  - LLM self-evaluation is notoriously prone to over-confidence: models frequently attempt to answer complex legal or transactional disputes that require authenticated account access, hallucinating procedures rather than escalating.
  - A ML binary classifier acts as a black box whose decision boundaries shift unpredictably when retrained, making compliance auditing difficult.
  - Pure keyword rules fail on subtle complaints that describe disputes without using explicit trigger words.
- **Evidence/tradeoff:** The multi-signal policy achieved an **82.42% Auto-Handle Precision** and caught **76.12% of true human escalations**. The tradeoff was 58 false escalations (43.61% of solvable queries), which we accepted to ensure safety over blind automation.

---

### Decision 10: Zero-Leakage Threshold Grid Calibration on Validation Data

- **Decision:** Calibrated the escalation thresholds ($\tau_{\text{conf}} = 0.55, \tau_{\text{sim}} = 0.65, N_{\text{evidence}} = 1$) via an exhaustive grid search exclusively on the 2,906-sample validation split, leaving the 200-sample Golden Set completely quarantined until final benchmark evaluation.
- **Why:** Tuning decision thresholds on test data introduces data snooping bias, invalidating the integrity of the benchmark and exaggerating production performance.
- **Alternatives considered:**
  1. Tuning thresholds directly on the 200 Golden Set cases.
  2. Using uncalibrated default rules of thumb (e.g., $\tau_{\text{conf}} = 0.50, \tau_{\text{sim}} = 0.50$).
  3. Setting aggressive thresholds to maximize the headline automation rate ($> 70\%$).
- **Why we rejected them:**
  - Tuning on the Golden Set violates fundamental machine learning principles and would result in an overfitted, overly optimistic safety assessment.
  - Uncalibrated defaults resulted in extreme failure modes: either 0% automation (escalating everything) or a 45% false auto-handle rate (endangering safety).
  - Maximizing automation produced an unacceptable safety profile, missing over 45% of critical human escalations.
- **Evidence/tradeoff:** The validation grid search identified the **Conservative Policy** as optimal for enterprise risk management. Evaluated blindly on the Golden Set, it maintained an 82.42% precision, confirming that validation tuning generalized to unseen test cases without degradation.

---

### Decision 11: LLM Engine Choice — Vertex AI Gemini 2.5 Flash via Native Google GenAI SDK

- **Decision:** Selected `gemini-2.5-flash` on Google Cloud Vertex AI using the native `google-genai` SDK (`from google import genai`) with strict structured JSON schema enforcement (`response_mime_type="application/json"`).
- **Why:** `gemini-2.5-flash` offers an optimal combination of low latency (~1.2s per generation), enterprise SLAs, GCP security governance, low token cost, and native Pydantic schema validation. Using the modern SDK ensured forward compatibility and avoided deprecated wrappers.
- **Alternatives considered:**
  1. Self-hosting an open-source model (Llama-3-8B or Mistral-7B via vLLM).
  2. Commercial APIs from other providers (OpenAI GPT-4o-mini, Anthropic Claude 3.5 Haiku).
  3. The legacy `google-generativeai` package.
- **Why we rejected them:**
  - Self-hosting open-source models required managing dedicated GPU instances (costing $500+/month), load balancing, and cold-start autoscaling.
  - Third-party APIs introduced multi-cloud vendor management and credentials outside the designated GCP environment.
  - The legacy `google-generativeai` SDK has been deprecated in favor of the unified `google-genai` client.
- **Evidence/tradeoff:** Gemini 2.5 Flash achieved **100% valid JSON schema compliance across all 200 golden test cases**, with zero parsing exceptions and an average roundtrip latency of 1.28 seconds.

---

### Decision 12: Prompt Architecture — Strict 4-Way Role Separation & Anti-Hallucination Guardrails

- **Decision:** Designed a generation prompt with four clearly demarcated XML-tagged blocks:
  1. `<system_instructions>`: Brand persona (`@AmazonHelp`), tone constraints (concise, professional, empathetic, Twitter character limits).
  2. `<customer_inquiry>`: Raw customer message and predicted intent label.
  3. `<retrieved_evidence>`: Up to 3 historical resolution pairs formatted with conversation IDs and resolution summaries.
  4. `<negative_constraints>`: Explicit prohibitions ("Do NOT invent return windows", "Do NOT cite unverified URLs", "Do NOT make financial promises").
- **Why:** LLMs conditioned on raw, unstructured prompts frequently hallucinate external corporate policies, mix customer complaints with brand statements, or invent fake customer service phone numbers. Clear role boundaries force the model to treat retrieved precedents as the sole source of operational truth.
- **Alternatives considered:**
  1. Free-form conversational few-shot prompting.
  2. Chain-of-thought prompting exposing internal reasoning steps to the customer.
  3. Retrieval-Augmented Generation (RAG) with entire uncompressed multi-turn conversation logs.
- **Why we rejected them:**
  - Free-form few-shot prompting occasionally caused the model to adopt customer complaints as factual statements.
  - Exposing chain-of-thought in customer service responses violates brand communication standards and Twitter character constraints.
  - Uncompressed multi-turn logs inflated prompt tokens by 400%, increasing latency without improving factual accuracy.
- **Evidence/tradeoff:** The negative-constraint prompt achieved a **4.70 / 5.00 Safety score** under LLM-as-a-judge and a **4.90 / 5.00 Safety score** under human expert review, with zero observed policy hallucinations across audited samples.

---

### Decision 13: Evaluation Methodology — 5-Dimensional LLM-as-a-Judge Rubric vs. BLEU/ROUGE

- **Decision:** Benchmarked response generation quality using an automated LLM-as-a-Judge (`gemini-2.5-flash`) scoring five distinct operational dimensions on a 1–5 scale with explicit qualitative rubrics:
  1. `correctness`
  2. `historical_grounding`
  3. `helpfulness`
  4. `brand_consistency`
  5. `safety_unsupported_claims`
- **Why:** Traditional n-gram overlap metrics (BLEU, ROUGE, METEOR) are ineffective for conversational support. A valid, empathetic paraphrasing that directs a customer to the correct help page can share zero n-grams with the historical tweet, scoring 0 BLEU despite being an excellent resolution. Conversely, repeating the customer's problem scores high ROUGE but provides zero customer help.
- **Alternatives considered:**
  1. Surface n-gram metrics (BLEU-4, ROUGE-L).
  2. A single holistic 1–5 score.
  3. Pairwise Elo win-rate comparison against baseline responses.
- **Why we rejected them:**
  - N-gram metrics penalize valid stylistic variations and fail to detect subtle factual hallucinations.
  - A single holistic score conflates tone with factual accuracy, making it impossible to diagnose whether a low score was caused by poor grammar or dangerous advice.
  - Pairwise Elo ranking is computationally quadratic ($O(N^2)$) and fails to provide absolute quality thresholds.
- **Evidence/tradeoff:** The 5-dimensional rubric successfully isolated specific model behaviors: revealing that while the agent achieved near-flawless Brand Consistency (4.65) and Safety (4.70), its Historical Grounding score (2.40) was penalized during escalations.

---

### Decision 14: Human vs. LLM Judge Validation — Independent Paired Correlation & Bias Audit

- **Decision:** Conducted an independent human expert validation across 20 representative test cases (100 paired evaluation ratings) using the exact same 5-dimensional rubric, calculating Pearson $r$, Spearman $\rho$, and Quadratic Weighted Kappa (QWK).
- **Why:** Validating an LLM pipeline solely with an LLM judge is circular and scientifically ungrounded. To rely on automated evaluation for regression testing, one must prove that the judge correlates with human judgment and explicitly identify its systematic biases.
- **Alternatives considered:**
  1. Accepting LLM judge scores as ground truth without human validation.
  2. Multi-model LLM jury (averaging scores from GPT-4o, Claude 3.5, and Gemini).
  3. Pure human evaluation across all 200 cases.
- **Why we rejected them:**
  - Unvalidated LLM judges suffer from prompt artifacts and cannot be trusted for safety certification.
  - An LLM jury multiplies API costs without eliminating shared model biases (e.g., all models tend to reward long, verbose, sycophantic replies).
  - Evaluating 200 cases across 5 dimensions with multiple human annotators is cost-prohibitive for rapid development iterations.
- **Evidence/tradeoff:** The paired validation proved strong human-judge alignment on **Correctness ($r = 0.9247$, QWK = 0.8201)** and **Helpfulness ($r = 0.8890$, QWK = 0.8772)**. Crucially, it revealed a **systematic severity bias of -0.85 points on Historical Grounding**, proving that the LLM judge unfairly penalized safe human escalations for not copying historical tracking links. This insight protected the engineering team from misdiagnosing a safe behavior as a system failure.

---

## 3. Summary of Decision Impacts & System Evolution

The table below summarizes the key architectural shifts and the resulting business and technical impacts:

| Component | Initial / Naive Approach | Adopted Engineering Decision | Primary Impact / Justification |
| :--- | :--- | :--- | :--- |
| **Brand Corpus** | Multibrand mixed pool | Focused `@AmazonHelp` corpus | Clean ground truth, uniform escalation boundary. |
| **Conversation Store** | Raw tweet pairs | Full thread reconstruction + resolution tagging | Eliminated toxic, unresolved, and truncated cases. |
| **Taxonomy** | 30+ granular micro-intents | 10 coarse pragmatic classes | High statistical support ($N>500$), 90.5% test accuracy. |
| **Test Benchmark** | Random 80/20 split | Stratified $N=200$ Golden Set with difficulty tiers | Exposed the 35% retrieval collapse on hard queries. |
| **Intent Classifier** | Deep transformer model | TF-IDF + Multinomial Logistic Regression | 0.3ms CPU latency, full feature interpretability. |
| **Dense Embeddings** | Commercial embedding API | Local `all-MiniLM-L6-v2` (384-d) | Sub-40ms CPU inference, zero API cost, 17.6 MB index. |
| **Vector Index** | Approximate ANN (HNSW) | Exact FAISS `IndexFlatIP` | Zero recall approximation error, 100% deterministic. |
| **Retrieval Top-k** | $k=1$ or $k=5$ | Fixed $k=3$ precedents | 81.0% Hit@3, optimal prompt token density. |
| **Escalation Engine** | LLM unconstrained decision | Multi-signal deterministic Python policy | 82.4% precision, full auditability, zero prompt injection. |
| **Threshold Tuning** | Test set tuning | Validation grid search (zero leakage) | Valid generalization, zero optimistic benchmark bias. |
| **LLM Backend** | Local open-source LLM | Vertex AI Gemini 2.5 Flash | 100% JSON schema compliance, 1.2s latency. |
| **Prompt Design** | Free-form few-shot | Strict 4-way XML negative constraints | 4.70/5.00 safety rating, zero policy hallucinations. |
| **Output Evaluation** | Surface BLEU/ROUGE | 5-dimensional LLM-as-a-Judge rubric | Granular diagnosis separating tone from factual safety. |
| **Judge Validation** | Unverified LLM scores | Paired human correlation ($r=0.8147$) | Detected and compensated for -0.85 grounding bias. |

---

## 4. Conclusion

These 14 decisions illustrate a consistent engineering philosophy: **prioritize deterministic safety, low latency, and auditability over unchecked complexity.** By selecting classical ML where appropriate (TF-IDF for 0.3ms routing), leveraging dense embeddings for semantic search (FAISS exact cosine), enforcing explicit deterministic guardrails for escalation, and grounding LLM generation in verified brand precedents, the system balances automated efficiency with operational safety.
