# Milestone 15: Honest Interpretation of Results & Evaluation Limitations

**Document Version:** 1.0  
**Effective Date:** 2026-09-10  
**Author:** AI Engineering & Evaluation Team (`@AmazonHelp` Support Pipeline)  
**Target Audience:** Executive Reviewers, Lead Machine Learning Engineers, Head of Customer Support Operations  
**Benchmark Artifacts:** `data/analysis/comprehensive_evaluation_harness.json`, `data/analysis/statistical_integrity_metrics.json`, `data/analysis/human_vs_llm_validation_results.json`, `data/processed/golden_evaluation_set.jsonl`

---

## 1. Executive Summary: The Intellectual Honesty Mandate

A hallmark of mature machine learning engineering is the ability to scrutinize one's own benchmarks with unsparing intellectual honesty. In high-stakes enterprise customer service—such as managing customer communications for `@AmazonHelp`—reporting flattering headline metrics without rigorous context is dangerous. A metric that looks outstanding in an offline notebook can mask catastrophic failure modes in live customer operations.

This document conducts a critical post-mortem of our hybrid support agent's headline performance. We do not artificially diminish the system’s genuine technical strengths (such as a **+76.50%** accuracy boost over baseline, deterministic sub-50ms retrieval, and zero policy hallucinations in audited samples). Nor do we downplay its structural vulnerabilities. 

Our explicit goal is to deconstruct what is misleading about our headline figures, quantify the statistical margins of error, and articulate the single strongest legitimate criticism of our system before enterprise deployment.

---

## 2. Master Headline Results Matrix

Below are the official headline metrics achieved by the `@AmazonHelp` Hybrid Support Agent across all project milestones:

| Metric Dimension | Official Headline Number | Point Estimate | Benchmark Baseline / Context | 95% Confidence Interval (Wilson) |
| :--- | :--- | :---: | :--- | :---: |
| **Intent Classification** | **Accuracy** | **90.50%** | Baseline 1 (Majority): 14.00% (+76.50% lift) | **[85.64%, 93.83%]** |
| | **Macro F1-Score** | **91.05%** | Baseline 1 (Majority): 2.46% (+88.59% lift) | *Unweighted average across 10 classes* |
| | **Weighted F1-Score** | **90.43%** | Evaluated on stratified Golden Set ($N=200$) | *Sample-weighted across 10 classes* |
| **Historical Retrieval** | **Top-1 Intent Hit Rate** | **65.00%** | FAISS `IndexFlatIP` on 12k resolved cases | **[58.16%, 71.27%]** |
| | **Top-3 Intent Hit Rate** | **81.00%** | Intent-congruent precedent within top 3 | **[75.00%, 85.83%]** |
| | **Top-5 Intent Hit Rate** | **85.00%** | Intent-congruent precedent within top 5 | **[79.39%, 89.29%]** |
| | **Mean Reciprocal Rank (MRR)**| **0.7289** | Average position of first relevant precedent | *Theoretical maximum: 1.000* |
| | **Mean Top-1 Cosine Sim.** | **0.7209** | Dense semantic embedding proximity | *384-d `all-MiniLM-L6-v2`* |
| **Escalation & Policy** | **Overall Policy Accuracy** | **63.00%** | Multi-signal calibrated decision engine | **[56.12%, 69.39%]** |
| | **Automation Rate** | **45.50%** | Inquiries routed to automated generation | **[38.75%, 52.42%]** |
| | **Escalation Safety Recall**| **76.12%** | True human escalation cases caught (51/67) | **[64.67%, 84.73%]** |
| | **Auto-Handle Precision** | **82.42%** | Auto-handled cases deemed safe (75/91) | **[73.33%, 88.88%]** |
| | **False Escalation Rate** | **43.61%** | Benign cases unnecessarily escalated (58/133) | **[35.48%, 52.10%]** |
| **LLM Response Quality**| **Overall Quality Score** | **3.57 / 5.00**| Evaluated via LLM-as-a-Judge ($N=20$) | *Scale: 1.00 (Poor) to 5.00 (Flawless)* |
| | **Brand Consistency** | **4.65 / 5.00**| Alignment with `@AmazonHelp` persona | *Extreme high clustering (std: 0.65)* |
| | **Safety / Hallucination** | **4.70 / 5.00**| Zero fabricated return policies or false claims | *Extreme high clustering (std: 0.78)* |
| | **Correctness** | **3.05 / 5.00**| Factual and procedural resolution accuracy | *Pearson $r = 0.9247$ with human experts* |
| | **Helpfulness** | **3.05 / 5.00**| Immediate actionability for customer | *Pearson $r = 0.8890$ with human experts* |
| | **Historical Grounding** | **2.40 / 5.00**| Direct derivation from retrieved precedent | *Severe negative rubric penalty on escalations*|
| **Human vs. LLM Alignment**| **Overall Correlation ($r$)**| **0.8147** | Pearson correlation across 100 paired scores | *$p = 1.2 \times 10^{-5}$ (Highly significant)* |
| | **Rank Correlation ($\rho$)**| **0.7648** | Spearman rank-order correlation | *$p = 8.6 \times 10^{-5}$* |
| | **LLM Judge Severity Bias** | **-0.31 points**| Mean delta (LLM Mean: 3.57 vs Human: 3.88) | *LLM judge is systematically harsher* |

---

## 3. Core Question: "What is Misleading About My Headline Number?"

If an executive reads only the headline summary:
> *"Our AI Support Agent achieves 90.50% Intent Classification Accuracy, an 81.00% Retrieval Hit Rate, 4.70/5 Safety, and automates 45.50% of customer inquiries,"*

they would conclude that the system is ready for immediate, unmonitored production deployment. 

**This conclusion is dangerously misleading.**

The headline numbers are misleading for three fundamental architectural reasons:

1. **Conflating Discrete Text Classification with Problem Resolution:**  
   Achieving 90.50% intent accuracy means the classifier correctly guessed a coarse category label (e.g., *"this tweet is about delivery"*). It does **not** mean the customer’s problem was understood, investigated, or solved. In customer service, routing an inquiry to the right bucket is merely step zero; the customer judges the brand on whether their missing parcel was located or their money returned.
2. **The Masking Effect of Aggregate Escalation Recall:**  
   A headline of "76.12% Escalation Safety Recall" sounds defensible. Inverted into operational terms, however, it reveals that **23.88% of high-risk, legally sensitive, or fraudulent customer disputes bypass human intervention entirely**. In a live deployment of 100,000 monthly inquiries, an 8.0% aggregate false auto-handle rate means **8,000 critical customer disputes** (alleging driver theft, forced deductions, or courier misconduct) would be answered by an automated bot giving generic tracking links.
3. **The Artificial Purity of the Curated Test Pool:**  
   The 90.50% accuracy was achieved on a hand-curated, stratified Golden Set ($N=200$) drawn exclusively from the **18.49%** of customer interactions that matched our 10-intent taxonomy. The remaining **81.51% of raw Twitter conversations**—consisting of compound requests, colloquial chatter, social mentions, and ambiguous rants—were excluded from evaluation. Claiming 90.5% accuracy on an uncurated enterprise social stream is an unvalidated extrapolation.

---

## 4. The Strongest Legitimate Criticism of Our System

Every engineering team must confront the sharpest counterargument to their work. The strongest legitimate criticism of our hybrid support agent is:

> ### **The Definitive Critique:**
> **"The Headline 90.5% Intent Accuracy and 45.5% Automation Rate create a dangerous illusion of production readiness by masking a 23.9% failure rate on critical human escalations and a 43.6% false-escalation penalty on routine inquiries, while evaluating only on an artificially balanced, single-turn, in-domain 18.5% slice of real-world Twitter traffic."**

### Why This Criticism Is Valid:
1. **The Safety Hazard (False Negatives):** In 16 out of 67 genuine escalation cases in our benchmark (such as `conv_206973`, where a driver attempted to force a package into a mailbox and walked away with it), the system gave a false `AUTO_HANDLE` decision. A customer experiencing courier misconduct who receives an automated canned response (*"Please track your order here"*) will experience immediate brand alienation and may escalate publicly or legally.
2. **The Economic Cost (False Positives):** Conversely, to catch 76% of escalations, our rigid static thresholds falsely escalated **43.61% of completely benign, solvable queries** (58 out of 133). Simple questions like *"what is your return window for mobile phones?"* or *"why does tracking say out for delivery at 7 PM?"* were pushed into human agent queues, creating artificial support backlog and negating the primary economic justification for building an automated bot.
3. **The Single-Turn Trap:** The benchmark measures only turn zero ($t=0$). Real customer support is an authenticated, stateful, multi-turn interaction. Evaluating single responses in isolation ignores the customer's inevitable follow-up when the self-service link does not resolve their underlying issue.

---

## 5. Deep Dive: The 11 Critical Evaluation Dimensions

To provide complete transparency, we examine our evaluation methodology across 11 structural dimensions.

---

### 1. Class Imbalance & Natural Frequency Re-weighting

In our benchmark design, the Golden Evaluation Set ($N=200$) was curated with **stratified balance**: every intent was allocated between 15 and 28 samples (a ratio of only $1.87 : 1$ between the largest and smallest class). 

However, in the natural corpus of 79,665 clean `@AmazonHelp` conversations (of which 14,729 matched our taxonomy), the distribution is intensely skewed:
- `ORDER_DELIVERY_AND_TRACKING` (3,686 cases) and `REFUND_STATUS_AND_DISPUTES` (3,108 cases) account for **46.13%** of all customer volume.
- In contrast, `RETURNS_AND_EXCHANGES` (541 cases) and `TECHNICAL_AND_PLATFORM_ISSUES` (546 cases) represent only **3.67%** and **3.71%** of volume (a natural ratio of $6.81 : 1$).

#### Quantitative Impact:
When we re-weight the per-intent F1 scores by their true natural ambient frequencies rather than the artificial golden-set weights, our headline classification metric shifts:
- **Macro F1 (Unweighted Average across 10 classes):** **91.05%**
- **Golden-Weighted F1 (Curated Split):** **90.43%**
- **Natural Frequency-Weighted F1 (Real-World Ambient):** **89.25%**

$$\Delta = -1.80\% \text{ absolute drop (falling below the psychological } 90\% \text{ threshold)}$$

**Why this matters:** Our lowest-performing intent is `REFUND_STATUS_AND_DISPUTES` (F1 score of **78.43%**). Because refunds represent nearly 21% of real-world volume, but only 13% of our golden set, the headline macro F1 score of 91.05% artificially inflates the system's day-to-day operational competence.

---

### 2. Easy vs. Difficult Examples: The "Difficulty Cliff"

During Milestone 5 curation, the 200 Golden Set cases were partitioned into three difficulty tiers:
- **Easy:** 90 cases (45%) — single clear problem, standard vocabulary, unambiguous intent.
- **Medium:** 70 cases (35%) — conversational phrasing, minor typos, mild frustration.
- **Hard:** 40 cases (20%) — compound problems, severe frustration, causal chains, missing context.

Comparing retrieval and escalation performance across these tiers reveals a **performance collapse** on complex cases:

| Metric | Easy Tier ($N=90$) | Medium Tier ($N=70$) | Hard Tier ($N=40$) | Difficulty Collapse (Hard vs. Easy) |
| :--- | :---: | :---: | :---: | :---: |
| **Retrieval Hit@1** | 72.22% | 71.43% | **37.50%** | **-34.72% absolute drop** |
| **Retrieval Hit@3** | 87.78% | 88.57% | **52.50%** | **-35.28% absolute drop** |
| **Retrieval Hit@5** | 88.89% | 92.86% | **62.50%** | **-26.39% absolute drop** |
| **Mean Reciprocal Rank (MRR)**| **0.7935** | **0.7988** | **0.4612** | **-0.3323 collapse** |
| **Escalation Safety Recall** | 75.00% | 91.30% | **60.00%** | **-15.00% drop (40% missed!)** |
| **Auto-Handle Precision** | 86.67% | 92.00% | **61.90%** | **-24.77% drop (38% unsafe)** |

**The Reality:** On Twitter, customers rarely tweet routine questions that are answered on the homepage; they reach out on social media precisely *because* standard self-service channels failed, meaning an enterprise Twitter queue is disproportionately loaded with **Hard** cases. If the live distribution contains 40% Hard cases rather than our curated 20%, the true retrieval Hit@3 will drop from **81.0% to ~70%**, and escalation safety recall will degrade significantly.

---

### 3. Rare Intents & The Illusion of 100% Recall

In our evaluation harness, the classifier achieved flawless **100% recall** on three distinct intent classes:
- `ORDER_CANCELLATION` (20/20 recall, F1: 95.24%)
- `RETURNS_AND_EXCHANGES` (15/15 recall, F1: 93.75%)
- `TECHNICAL_AND_PLATFORM_ISSUES` (15/15 recall, F1: 93.75%)

**The Statistical Caveat:**  
A score of 100% recall on 15 samples does not indicate perfection; it reflects **statistical sample starvation**. With $N=15$, each sample carries a massive **6.67% weight**. A single false negative would drop recall from 100% to 93.3%; two errors would drop it to 86.7%. 

More importantly, 15 samples cannot capture the diverse long-tail permutations of technical failures (e.g., Prime Video DRM errors on smart TVs, Fire Stick boot loops, Alexa multi-room audio glitches, regional geoblocking). Claiming our system has "solved" technical platform issues based on 15 test examples is an overstatement.

---

### 4. Dataset Sampling & Survivorship / Resolution Bias

Our Historical Knowledge Base of 12,000 cases was constructed by filtering for conversations with explicit resolution tags (`resolved_guidance_provided` or `resolved_customer_confirmed`).

**The Survivorship Distortion:**  
By indexing only successfully resolved cases, the vector knowledge base suffers from **survivorship bias**:
1. It omits historical cases where the customer gave up in frustration, where the issue was dropped, or where the brand's response was unhelpful.
2. It systematically over-represents issues that have clean, public, canned resolutions (e.g., standard tracking URLs, return links) and under-represents intractable disputes (e.g., courier delivery disputes, damaged high-value electronics) that were handled via phone calls or private direct messages.

Consequently, dense retrieval easily finds precedents for routine issues, but encounters an artificial knowledge desert when a customer presents an ambiguous or edge-case complaint.

---

### 5. Golden-Set Sample Size ($N=200$) & Statistical Confidence Intervals

While $N=200$ is a standard, labour-intensive benchmark size for qualitative customer support auditing, its binomial sampling error is wide.

Using the **Wilson score interval** ($\alpha = 0.05$), we calculate the 95% confidence intervals for our core metrics:

```
                          95% STATISTICAL CONFIDENCE SPANS
                               (Wilson Score Interval)
                               
Intent Accuracy (90.5%)     ├──[85.6%]──────●──────[93.8%]──┤ (Span: 8.19%)
Retrieval Hit@3 (81.0%)     ├────[75.0%]─────●──────[85.8%]──┤ (Span: 10.83%)
Automation Rate (45.5%)     ├───[38.8%]─────●──────[52.4%]───┤ (Span: 13.67%)
Safety Recall (76.1%)       ├─────[64.7%]────●───────[84.7%]─┤ (Span: 20.06%)
False Auto-Handle (23.9%)   ├───[15.3%]─────●───────[35.3%]──┤ (Span: 20.06%)
```

#### Critical Observations:
- **Intent Accuracy:** Our true population accuracy is reliably estimated within **[85.64%, 93.83%]**.
- **Escalation Safety Recall:** For our safety-critical metric ($N=67$ true escalations), the 95% confidence interval spans **[64.67%, 84.73%]**—a range of over **20 percentage points**.
- **False Auto-Handle Rate:** The true proportion of high-risk complaints mishandled by the bot could be as high as **35.33%** in the general population.

Reporting a single point estimate of "76.1% safety recall" conceals the statistical reality that the system's safety margin could be substantially lower in production.

---

### 6. Temporal Distribution & Policy Stationarity

The customer service interactions in `twcs.csv` date from **October 2017 through December 2017** (with a few dating back to 2015). 

Evaluating a system in 2026 using 2017 data assumes **temporal stationarity**—an assumption that does not hold in consumer retail:
1. **Policy Evolution:** Return windows, holiday replacement policies, and warranty processes change annually. A 2017 historical precedent will cite obsolete policies (e.g., outdated return cutoff dates or legacy refund processing times).
2. **Infrastructure & UI Drift:** Historical agent responses frequently provide direct hyperlinks (`https://t.co/...` pointing to Amazon help pages). In 2026, many of these URLs are dead links or redirect to redesigned web portals.
3. **Product Catalog Changes:** In 2017, Alexa, Echo Dot 2nd Gen, and Fire TV Stick 1080p dominated hardware queries; modern support requests involve Echo Show 15, Kindle Scribe, Luna cloud gaming, and newer Prime Video app architectures.

Offline evaluation on static historical data cannot verify whether the system’s guidance remains procedurally correct today.

---

### 7. Ambiguity & The Causal Chain Fallacy

Traditional NLP benchmarks treat intent classification as a mutually exclusive, single-label problem. In live customer support, inquiries are frequently **causal chains**:

$$\text{Problem: Delivery Delayed (Logistics)} \longrightarrow \text{Action: Cancel Order (Logistics)} \longrightarrow \text{Remedy: Refund Demanded (Finance)}$$

Consider real golden-set example `conv_319695`:
> *"kindly track order no #4__credit_card__ neither product nor refund received even after a month"*

Is this `ORDER_DELIVERY_AND_TRACKING` or `REFUND_STATUS_AND_DISPUTES`?
- Ground truth assigned it to `ORDER_DELIVERY_AND_TRACKING`.
- The classifier assigned it to `REFUND_STATUS_AND_DISPUTES` (83.5% confidence).
- In our failure analysis (Milestone 14), **57.89% of all classification errors** (11 out of 19) were boundary disputes between delivery logistics and financial refunds.

Calling this a "classifier failure" is partly an artifact of forcing a multi-stage customer journey into a single discrete bucket. The system failed because the taxonomy assumes single-label purity.

---

### 8. Retrieval Limitations: Intent Hit Rate vs. Resolution Utility

Our headline retrieval metric is **Hit@3 = 81.00%**. 

#### What Hit@3 Actually Measures:
Hit@3 measures whether *at least one* of the top 3 retrieved cases shares the **same intent label** as the query. It does **not** measure whether the retrieved precedent contains the specific facts, links, or procedures required to solve the customer's exact inquiry.

#### Empirical Evidence of Retrieval Gaps:
- **Top-1 Intent Relevance:** In **35.00%** of cases, the single best retrieved historical case belonged to an irrelevant intent category.
- **Top-3 All-Relevant Rate:** In only **64.17%** of cases were all 3 retrieved precedents relevant to the query topic.
- **Lexical Dilution:** Dense sentence embeddings (`all-MiniLM-L6-v2`) pool tokens uniformly. When a customer writes: *"I buy everything through @115830 Had to return 1 item now i want to return another but my printer is broken.......no one is helping"*, the dense model retrieves general complaints about poor service, completely missing the key physical constraint: **printer is broken / paperless return required**.

An 81% intent hit rate does not translate to an 81% solution retrieval rate.

---

### 9. LLM-as-a-Judge Limitations & Rubric Blind Spots

In Milestone 12, our LLM judge (`gemini-2.5-flash`) scored 20 representative responses across 5 dimensions, producing an overall average of **3.57 / 5.00**.

However, a close examination of the judge's scorecards reveals two major evaluation distortions:

1. **The Escalation Grounding Penalty:**  
   The judge awarded **Historical Grounding** a low average score of **2.40 / 5.00**. A qualitative audit revealed that whenever the agent correctly chose `ESCALATE` and generated a safe, courteous handoff message:
   > *"To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team..."*
   the LLM judge awarded a Grounding score of **1 or 2**, justifying the deduction with: *"The reply does not incorporate the specific tracking URL or direct resolution steps shown in the retrieved historical precedent."*  
   The judge penalized the model for escalating safely instead of recklessly copying unauthenticated historical actions.
2. **Ceiling Effects & Compressed Variance on Safety and Brand:**  
   The judge awarded **Brand Consistency** an average of **4.65** ($\sigma = 0.65$) and **Safety** an average of **4.70** ($\sigma = 0.78$). Across 20 evaluations, the judge never used a score below 3 on Safety, exhibiting a pronounced positivity bias for polished corporate prose.

---

### 10. Human vs. LLM Judge Disagreement

In Milestone 13, independent human experts evaluated the exact same 20 responses using an identical rubric. Comparing the paired scores highlights where automated evaluation diverges from human operational judgment:

| Evaluation Dimension | Pearson Correlation ($r$) | Quadratic Weighted Kappa (QWK) | Mean LLM Score | Mean Human Score | Systematic Bias ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Correctness** | **0.9247** | **0.8201 (Excellent)** | 3.05 | 3.65 | -0.60 (Harsher) |
| **Historical Grounding**| **0.5635** | **0.4533 (Moderate)** | 2.40 | 3.25 | **-0.85 (Severe Penalty)** |
| **Helpfulness** | **0.8890** | **0.8772 (Excellent)** | 3.05 | 3.00 | +0.05 (Neutral) |
| **Brand Consistency** | **0.6820** | **0.6753 (Substantial)**| 4.65 | 4.60 | +0.05 (Neutral) |
| **Safety / Unsupported**| **0.5121** | **0.3243 (Fair/Poor)** | 4.70 | 4.90 | -0.20 (Harsher) |
| **Overall Score** | **0.8147** | **0.6259 (Substantial)**| **3.57** | **3.88** | **-0.31 (Harsher)** |

#### Key Insights:
- While **Correctness** and **Helpfulness** show strong correlation ($r \approx 0.90$), **Historical Grounding** achieves an inter-rater agreement of only **QWK = 0.4533**.
- Human evaluators recognized that a polite escalation is the procedurally correct way to handle sensitive issues, scoring it 3 or 4. The LLM judge mechanically demanded lexical overlap with the prompt, scoring it 1 or 2.
- Automated LLM judge metrics cannot be accepted as ground truth without human calibration.

---

### 11. Offline Benchmark vs. Real-Time Production Support

The final and most significant source of evaluation divergence is the fundamental gap between an **offline batch benchmark** and **live customer support operations**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      THE PRODUCTION REALITY WEDGE                           │
├──────────────────────────────────────┬──────────────────────────────────────┤
│       OFFLINE LABORATORY BENCHMARK   │       LIVE ENTERPRISE PRODUCTION     │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • Single-turn evaluation (t = 0)     │ • Multi-turn stateful dialogue       │
│ • Static text-only prompt input      │ • Live CRM / Order Tracking API auth │
│ • Curated 10-intent closed world     │ • Uncurated open-world social stream │
│ • 18.49% matched sample subset       │ • 81.51% unclassified noisy chatter  │
│ • Generic tracking links accepted    │ • Live package location demanded     │
│ • Zero state or context tracking     │ • Authentication & session management│
│ • Infinite retry / zero time SLA     │ • Strict real-time latency (<1.5s)   │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

#### The 18.5% In-Domain Horizon:
Our evaluation is conducted exclusively on conversations that cleanly matched our 10-class intent taxonomy (14,729 out of 79,665 clean conversations, or **18.49%**). 

The remaining **81.51% of customer messages** in the dataset consist of unstructured social mentions, multi-topic complaints, brand banter, and ambiguous inquiries. Deploying our model directly to Twitter would expose it to a distribution where over 80% of incoming messages fall outside its evaluated competency.

---

## 6. Synthesis: What We Learned

Reviewing these 11 dimensions reinforces an essential principle: **a machine learning model is not a production system.**

Our technical accomplishments remain genuine:
1. The classical ML baseline established an intent classification foundation (+76.5% over majority rule).
2. The FAISS vector retrieval engine surfaces relevant brand precedent in 42 milliseconds.
3. The generation pipeline prevents hallucinations, achieving high safety scores under both human and automated evaluation.
4. The explicit escalation engine automates 45.5% of volume while maintaining an 82.4% safety precision.

However, the headline metrics must be interpreted with appropriate context:
- True intent accuracy in an open-world distribution will be closer to **~80–85%**, not 90.5%.
- Escalation safety recall on hard cases drops to **60.0%**, leaving an operational safety gap that requires stronger guardrails.
- Automation is bounded at **~45%**; attempting to force higher automation without live database integrations results in hallucinated or unhelpful bot answers.

---

## 7. Actionable Roadmap: Bridging the Production Gap

To transition this hybrid prototype into a production-ready enterprise agent, we recommend five engineering priorities:

1. **Replace Static Regexes with a Semantic Safety Classifier:**  
   Upgrade the escalation policy from keyword checks to a fine-tuned binary safety model (e.g., DeBERTa-v3) that evaluates risk and operational conflict directly from customer narrative context.
2. **Implement Intent-Tiered Risk Thresholds:**  
   Replace universal confidence thresholds with tiered gates: lower the automation barrier (e.g., confidence $\ge 0.45$) for low-risk informational queries (tracking procedures, return policies) to recapture the 58 falsely escalated cases, while maintaining strict criteria ($\ge 0.75$) for financial and account security inquiries.
3. **Deploy Hybrid Retrieval (Dense FAISS + Sparse BM25):**  
   Mitigate dense retrieval dilution by combining sentence embeddings with exact-token BM25 matching, partitioned by predicted intent.
4. **Integrate Real-Time Backend APIs:**  
   Connect the generation prompt to authenticated mock order APIs, enabling the model to state actual package coordinates rather than generic tracking links.
5. **Establish a Continuous Human-in-the-Loop Audit:**  
   Maintain automated daily correlation audits between human supervisors and LLM judges, recalibrating scoring rubrics against live customer satisfaction (CSAT) scores.
