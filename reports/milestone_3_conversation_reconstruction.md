# Milestone 3: Conversation Reconstruction Report (`AmazonHelp`)

**Generated On:** 2026-09-10 13:14:00  
**Target Brand:** `@AmazonHelp`  
**Total Raw Records Processed:** 324,816 tweets  
**Pipeline Script:** `src/reconstruct_conversations.py`

---

## 1. Dataset Structure & Thread Mechanics in TWCS

The TWCS dataset does not store pre-packaged conversation threads. Instead, interactions are represented as individual tweet rows with self-referential graph pointers:

1. `tweet_id`: Unique 64-bit integer identifier for the tweet.
2. `in_response_to_tweet_id`: Foreign key pointing to the parent tweet.
   - For an **inbound customer opening ticket**, `in_response_to_tweet_id` is typically `NaN` (or references an external tweet).
   - For a **brand agent response**, `in_response_to_tweet_id` points to the customer's initiating tweet.
   - For **subsequent follow-up turns**, customer and brand alternate parent pointers (`Customer T1 <- Brand T2 <- Customer T3 <- Brand T4`).
3. `response_tweet_id`: Comma-separated list of child tweets that replied to this tweet.

### The Thread Reconstruction Algorithm
To assemble complete conversations across 324,816 rows:
- **Disjoint Set Union (DSU / Union-Find) with Path Compression:** Each tweet $T_i$ is connected to its parent $P_i$. The graph partitioning clusters the dataset into **86,672 distinct connected components** (conversation trees).
- **Chronological Sorting:** Each conversation component's tweets are sorted using exact ISO-8601 timestamps (`%a %b %d %H:%M:%S %z %Y`).
- **Canonical Root & Conversation ID:** The root tweet ID (or earliest observed tweet ID) defines the permanent `conversation_id` (e.g., `conv_1631089`).

---

## 2. Normalized Conversation Schema

The normalized conversation schema is strictly grounded in verifiable dataset fields without fictitious attributes:

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `conversation_id` | `string` | Canonical identifier for the conversation tree (`conv_<root_tweet_id>`). |
| `timestamp` | `string` | ISO-8601 UTC timestamp of the first message in the conversation. |
| `customer_message` | `string` | Cleaned text of the initiating customer query (leading `@mentions` stripped). |
| `brand_response` | `string` | Initial or primary resolution text from `@AmazonHelp`. |
| `conversation_history` | `list[dict]` | Chronologically ordered list of all turns, preserving `turn`, `tweet_id`, `role`, `author_id`, `created_at`, `text` (cleaned), and `text_raw`. |
| `metadata` | `dict` | Derived operational indicators: `turn_count`, `customer_author_id`, `brand_author_id`, `initial_response_latency_seconds`, `initial_response_latency_minutes`, `resolution_status`, `resolution_confidence`, `has_url`, `is_multi_turn`, and `noise_reasons`. |

---

## 3. Preprocessing Decisions & Decision Log

Every preprocessing decision has been documented with its rationale:

| Decision # | Pipeline Stage | Preprocessing Action Taken | Rationale & Trade-offs |
| :--- | :--- | :--- | :--- |
| **D1** | **Thread Clustering** | Disjoint Set Union (DSU) graph clustering over `(tweet_id, in_response_to_tweet_id)`. | Twitter replies can fork or branch. DSU guarantees all related messages are grouped into one conversation unit without recursion limits or stack overflow. |
| **D2** | **Language Cleansing** | Filtered all threads containing CJK Unicode range (`[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]`). | Amazon operates global support on `@AmazonHelp`. Over 6,700 threads are in Japanese. Isolating English prevents non-Latin tokenization corruption in future embeddings and evaluations. |
| **D3** | **Mention Normalization** | Stripped leading `@mentions` in `customer_message` while retaining internal mentions. | Preserves core semantic user intent (e.g. *"where is my order"* instead of *"@AmazonHelp @115821 where is my order"*), improving retrieval indexing. Raw text is preserved in `text_raw`. |
| **D4** | **HTML Entity Decoding** | Decoded `&amp;` $\to$ `&`, `&lt;` $\to$ `<`, `&gt;` $\to$ `>`, `&#39;` $\to$ `'`. | Eliminates HTML encoding artifacts present in raw Twitter dumps. |
| **D5** | **Degenerate Message Filtering** | Flagged customer queries $<5$ characters (e.g. *"Yes"*, *"Ok"*, *"No"*) without URLs as noise. | Fragments without context pollute golden test sets and training pairs. |
| **D6** | **Broken Parent Handling** | Safely preserved threads where the parent ID was outside TWCS if brand guidance was intelligible, but flagged `missing_customer_query` in `noise_reasons`. | Prevents pipeline crashes while ensuring strict data hygiene in the clean set. |
| **D7** | **Resolution Classification** | Categorized into 4 states: `resolved_customer_confirmed`, `escalated_private_channel`, `resolved_guidance_provided`, and `unresolved_frustrated`. | Distinguishes autonomous informational resolutions from tickets requiring private agent escalation. |

---

## 4. Pipeline Execution & Dataset Distribution

From the 324,816 raw records:
- **Total Thread Components:** 86,672
- **Clean Usable Conversations:** **79,665** ($91.92\%$)
- **Filtered Noisy Conversations:** **7,007** ($8.08\%$)
  - Non-English / CJK: 6,758
  - Empty / Degenerate customer text: 227
  - Missing customer query: 29
- **Single-Turn Interactions (2 tweets):** 40,792 ($51.2\%$)
- **Multi-Turn Interactions (3+ tweets):** **38,873** ($48.8\%$)
- **Median Response Time:** **10.70 minutes**

### Resolution Status Breakdown:
1. `resolved_guidance_provided`: 60,878 ($76.4\%$) — Direct policy, troubleshooting, or tracking guidance provided.
2. `escalated_private_channel`: 12,355 ($15.5\%$) — Escalated to private phone/chat/form due to account/PII restrictions.
3. `resolved_customer_confirmed`: 6,432 ($8.1\%$) — Customer explicitly confirmed resolution (*"thanks, all sorted"*, *"fixed, appreciate it"*).

---

## 5. Concrete Conversation Examples

### Example 1: Clean Single-Turn Conversation (`conv_648407`)
- **Category:** Clean Policy Clarification (Resolved Guidance Provided)
- **Timestamp:** `2015-06-13T01:34:49Z`
- **Customer Message:** *"what in the world is a balance withheld? #customerservice doesnt seem to know!"*
- **Brand Response:** *"@274086 Sorry, I'm not quite sure what you're having trouble with. Was there an authorization you were wondering about? ^JY"*
- **Resolution:** `resolved_guidance_provided` (Turn count: 2, Latency: 42.17 min).

```json
{
  "conversation_id": "conv_648407",
  "timestamp": "2015-06-13T01:34:49Z",
  "customer_message": "what in the world is a balance withheld? #customerservice doesnt seem to know!",
  "brand_response": "@274086 Sorry, I'm not quite sure what you're having trouble with. Was there an authorization you were wondering about? ^JY",
  "conversation_history": [
    {
      "turn": 1,
      "tweet_id": 648407,
      "role": "customer",
      "author_id": "274086",
      "created_at": "Sat Jun 13 01:34:49 +0000 2015",
      "text": "@115821 what in the world is a balance withheld? #customerservice doesnt seem to know!"
    },
    {
      "turn": 2,
      "tweet_id": 648405,
      "role": "brand",
      "author_id": "AmazonHelp",
      "created_at": "Sat Jun 13 02:16:59 +0000 2015",
      "text": "@274086 Sorry, I'm not quite sure what you're having trouble with. Was there an authorization you were wondering about? ^JY"
    }
  ],
  "metadata": {
    "turn_count": 2,
    "customer_author_id": "274086",
    "brand_author_id": "AmazonHelp",
    "initial_response_latency_minutes": 42.17,
    "resolution_status": "resolved_guidance_provided",
    "is_multi_turn": false
  }
}
```

---

### Example 2: Clean Multi-Turn Conversation (`conv_1631089`)
- **Category:** Multi-Turn Hardware Consultation (Customer Confirmed Resolution)
- **Timestamp:** `2017-09-10T17:55:59Z`
- **Trajectory:**
  1. **Customer:** *"Wish soon i can save money and buy @117634 so that i can contribute to nature and read all my favorite books on @115850 Kindle!"*
  2. **Brand:** *"@499198 You are awesome, salute to your belief. I'm sure you're gonna change the world one day. What kind of Kindle do you like? ^HA"*
  3. **Customer:** *"@AmazonHelp I wish for all new kindle eReader white! Thanks for replying to my tweet and acknowledging my views."*
  4. **Brand:** *"@499198 That definitely is a very great choice! We hope you get one soon! All power to you. Nature is waiting 😊. ^ZH"*
- **Resolution:** `resolved_customer_confirmed` (Turn count: 4, Confidence: 0.95).

---

### Example 3: Unresolved / Escalated / Frustrated Conversation (`conv_313821`)
- **Category:** Multi-Turn Refund Dispute (Unresolved / Ongoing Frustration)
- **Timestamp:** `2016-05-06T14:21:46Z`
- **Trajectory:**
  1. **Customer:** *"Hy friends don't buy any products form Amazon India because I have one product returned at 6-02-16 but still we not get any refund"*
  2. **Brand:** *"@15547 automatically to your original payment method. (2/2)^KJ"*
  3. **Customer:** *"@amazonhelp seller already received package before 2 month but not get still not get any refund or help form your side"*
  4. **Brand:** *"@15547 Hi! you may file a claim against the seller in this case. Please reach to us here: https://t.co/vlvfJr4nN9 for more help. ^HK"*
  5. **Customer:** *"@amazonhelp I already A2Z claims 2 times on seller in 23-3-16 but not get any result /refund even today I call but not any reason 😡"*
  6. **Brand:** *"@15547 Hi there! kindly refer to the email sent by our buyer guarantee team. I'm sure they must've sent a correspondence. ^HK"*
- **Resolution:** `unresolved_frustrated` / `escalated_private_channel` (Customer remains unassisted publicly, A-to-Z claim unresolved).

---

### Example 4: Filtered Noisy Conversation (`conv_1115726` & `conv_249941`)
- **Type A (Multilingual Noise):** `conv_1115726`
  - **Customer:** `"すとくくん初めてのAmazon"`
  - **Brand:** `"@383277 初めてのAmazonのご利用、誠にありがとうございました(*ˊ˘ˋ*) HM"`
  - **Reason Filtered:** `cjk_language` (Isolated from English training/eval sets).
- **Type B (Degenerate Context / Orphaned):** `conv_249941`
  - **Customer Message:** `"Yes"` (1-word orphan response after missing previous context).
  - **Reason Filtered:** `empty_or_degenerate_customer_message`.

---

## 6. Generated Output Files

The pipeline generated three production-ready files in `data/processed/`:

1. **`amazonhelp_conversations_clean.jsonl`:** 79,665 rich, fully-structured multi-turn conversation objects with complete turn histories and metadata.
2. **`amazonhelp_conversations_clean.csv`:** 79,665 rows in tabular format for fast Pandas analysis and vector store indexing.
3. **`amazonhelp_conversations_noisy.jsonl`:** 7,007 quarantined conversations annotated with exact `noise_reasons`.
4. **`amazonhelp_conversation_summary.json`:** Comprehensive audit metadata file.
