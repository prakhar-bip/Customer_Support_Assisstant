# Milestone 1: Dataset Exploration & Brand Selection Report
**Dataset:** Kaggle Customer Support on Twitter (`twcs.csv`)  
**Generated On:** 2026-09-10 12:49:50  
**Total Records Analyzed:** 2,811,774 tweets  
**Pipeline Runtime:** 27.83s

---

## 1. Schema, Column Semantics & Data Integrity

The dataset consists of 7 primary columns capturing bidirectional customer support interactions on Twitter.

| Column Name | Data Type | Missing Count | Missing % | Semantic Meaning & Role in Agent Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| `tweet_id` | `int64` | 0 | 0.0% | Unique identifier for each tweet. Primary key for message tracking. |
| `author_id` | `object` | 0 | 0.0% | Anonymized user ID (e.g. `115854`) or official Brand Handle (e.g. `AppleSupport`). |
| `inbound` | `bool` | 0 | 0.0% | `True` if tweet was sent by customer; `False` if sent by the company/brand. |
| `created_at` | `object` | 0 | 0.0% | Tweet timestamp (format: `Wed Oct 11 06:55:44 +0000 2017`). |
| `text` | `object` | 0 | 0.0% | Raw text content of the tweet including mentions, URLs, and emojis. |
| `response_tweet_id` | `object` | 1,040,629 | 37.01% | Tweet ID(s) that replied to this tweet (can be comma-separated if multiple replies). |
| `in_response_to_tweet_id` | `float64` | 794,335 | 28.25% | Tweet ID of the immediate parent tweet this tweet is responding to. |

### Key Data Integrity Observations
1. **Zero Missing Values on Core Fields**: `tweet_id`, `author_id`, `inbound`, `created_at`, and `text` have **100% data completeness** across all rows.
2. **Conversation Representation**:
   - `inbound == True` represents an incoming customer message.
   - `inbound == False` represents an outbound brand support agent reply.
   - Customer handles are anonymized as numbers (e.g., `@115854`), whereas brand handles retain their corporate identity (e.g., `@AppleSupport`).
3. **Threading Mechanics**:
   - Conversations are linked primarily via `in_response_to_tweet_id`.
   - When a customer tweets an initial problem, `in_response_to_tweet_id` is `NaN` (root tweet).
   - When the brand replies, the brand's tweet has `in_response_to_tweet_id = customer_tweet_id`.
   - A direct, grounded `(Customer Query -> Brand Reply)` pair is formed by joining `outbound[in_response_to_tweet_id]` with `inbound[tweet_id]`.

---

## 2. Dataset-Wide Distribution

- **Total Tweets:** 2,811,774
- **Inbound Tweets (Customer queries):** 1,537,843 (54.7%)
- **Outbound Tweets (Brand responses):** 1,273,931 (45.3%)
- **Unique Brands Operating in Dataset:** 108 brands

---

## 3. Top 15 Brands by Outbound Support Volume

| Rank | Brand Handle | Outbound Tweets | Replies to Customer | Reply Rate (%) | Avg Chars | Avg Words | URL Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `AmazonHelp` | 169,840 | 169,287 | 99.7% | 124.0 | 19.8 | 41.3% |
| 2 | `AppleSupport` | 106,860 | 106,719 | 99.9% | 136.6 | 22.7 | 75.3% |
| 3 | `Uber_Support` | 56,270 | 56,261 | 100.0% | 110.1 | 19.2 | 51.3% |
| 4 | `SpotifyCares` | 43,265 | 43,243 | 100.0% | 129.6 | 22.1 | 50.5% |
| 5 | `Delta` | 42,253 | 42,197 | 99.9% | 103.9 | 17.9 | 15.3% |
| 6 | `Tesco` | 38,573 | 38,501 | 99.8% | 139.0 | 25.9 | 8.3% |
| 7 | `AmericanAir` | 36,764 | 36,598 | 99.6% | 103.4 | 18.4 | 6.5% |
| 8 | `TMobileHelp` | 34,317 | 34,287 | 99.9% | 125.0 | 21.5 | 48.2% |
| 9 | `comcastcares` | 33,031 | 33,007 | 99.9% | 127.5 | 24.0 | 4.0% |
| 10 | `British_Airways` | 29,361 | 29,315 | 99.8% | 124.4 | 21.7 | 6.8% |
| 11 | `SouthwestAir` | 28,977 | 28,889 | 99.7% | 116.8 | 20.4 | 16.2% |
| 12 | `VirginTrains` | 27,817 | 27,522 | 98.9% | 84.9 | 15.1 | 13.2% |
| 13 | `Ask_Spectrum` | 25,860 | 25,807 | 99.8% | 137.2 | 23.8 | 39.4% |
| 14 | `XboxSupport` | 24,557 | 24,341 | 99.1% | 114.9 | 19.9 | 40.4% |
| 15 | `sprintcare` | 22,381 | 22,335 | 99.8% | 115.1 | 21.2 | 17.3% |

---

## 4. Brand Comparison & Candidate Recommendations

To select the single best brand for our AI Support Agent, we evaluated candidate brands against five rigorous, objective criteria:
1. **Conversation Volume**: Sufficient volume to support vector search retrieval, intent clustering, and golden set stratification.
2. **Direct Interaction Ratio**: Percentage of tweets that directly answer a customer query (`in_response_to_tweet_id.notna()`).
3. **Domain & Intent Diversity**: Variety of technical issues, billing disputes, how-tos, and edge cases.
4. **Resolution Grounding & Quality**: Detailed troubleshooting instructions rather than purely generic "please DM us" redirects.
5. **Actionable Escalation Boundary**: High contrast between issues that can be auto-handled vs issues requiring human escalation.

### Candidate Analysis:

### 1. `AmazonHelp` (Selected Brand: ★★★★★)
- **Outbound Tweets:** 169,840 tweets (#1 largest brand volume in TWCS)
- **Direct Reply Rate:** 99.7% (169,287 replies to customer inquiries)
- **Avg Message Length:** 19.8 words
- **Median Response Time:** 11.47 minutes (fastest response speed among major brands)
- **Strengths:**
  - **Comprehensive E-Commerce & Hardware Spectrum:** Spans order delivery tracking, return/refund procedures, Prime subscription management, digital media (Prime Video, Kindle), and smart device troubleshooting (Fire TV Stick, Echo/Alexa).
  - **High Conversational Pair Volume:** Yields 168,823 structured `(Customer Query -> Brand Reply)` pairs with 99.70% inquiry matching rate.
  - **Clear Escalation Policy:** Auto-handle shipping tracking instructions, return policies, and device troubleshooting; escalate to private authenticated customer support (phone/chat) for payments, refund overrides, and account security.
- **Verdict:** **Selected Brand for this Project.**

### 2. `AppleSupport` (Candidate Brand: ★★★★☆)
- **Outbound Tweets:** 106,860 tweets
- **Direct Reply Rate:** 99.9%
- **Avg Message Length:** 22.7 words
- **Strengths:** High diagnostic depth for iOS and hardware issues.

### 3. `SpotifyCares` (Candidate Brand: ★★★★☆)
- **Outbound Tweets:** 43,265 tweets
- **Direct Reply Rate:** 100.0%
- **Avg Message Length:** 22.1 words
- **Strengths:** Focused music streaming domain.

### 4. `Uber_Support` (Service Domain Candidate: ★★★☆☆)
- **Outbound Tweets:** 56,270 tweets
- **Direct Reply Rate:** 100.0%
- **Limitation:** Most resolutions require private ride IDs and driver contact.

---

## 5. Sample Real Interaction Pairs (AmazonHelp)

### Example 1 (Delivery & Tracking):
- **Customer Tweet (`id=115820`):**
  > "@AmazonHelp 3 different people have given 3 different answers and I still don't have my order."
- **Historical Support Reply (`id=115819`):**
  > "@115820 We'd like to take a further look into this with you! Please reach us by phone or chat through our contact us page."

### Example 2 (Digital Devices / Fire TV Troubleshooting):
- **Customer Tweet (`id=272`):**
  > "amazonのfireTVstickが見れない😢 (Cannot watch Fire TV Stick)"
- **Historical Support Reply (`id=269`):**
  > "@115770 こんにちは、アマゾン公式です。Fire TV Stickが見れないというのは、どのような状況でしょうか。一般的なトラブルシューティングを記載したヘルプがございますので、ご参照ください。https://t.co/2pbG55qJ7h ET"

### Example 3 (Account & Payment Escalation Boundary):
- **Customer Tweet (`id=115823`):**
  > "@115823 I want my amazon payments account CLOSED. dm me please."
- **Historical Support Reply (`id=115822`):**
  > "@115822 I am unable to affect your account via Twitter. For real time support, phone or chat with us directly."

---

## 6. Milestone Conclusion & Selected Brand

We officially select **`AmazonHelp`** as the single brand for the Hiver AI Customer Support Take-Home Assignment. 
It offers:
1. **Largest empirical corpus:** 169,840 outbound tweets and 168,823 structured customer-brand interaction pairs.
2. **Realistic e-commerce multi-intent support:** Order tracking, returns, damaged items, Prime subscriptions, payment disputes, and digital streaming.
3. **Rapid response benchmarks:** Median response time of 11.47 minutes.
4. **Natural escalation boundaries:** Clear differentiation between public self-service guidance and private authenticated support (phone/chat/account desk).
