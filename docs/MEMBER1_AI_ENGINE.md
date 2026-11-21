# Member 1: AI engine (`app/ai/`)

One entry point: `orchestrator.handle_chat(ChatRequest) -> ChatResponse`.

## Pipeline for one message

| # | Step | File | LLM or code? |
|---|---|---|---|
| 1 | Load session memory; the session must belong to this `customer_id` | `memory.py` | code |
| 2 | Privacy and injection guard, run **before** any LLM call | `safety.py` | code |
| 3 | Detect language (en / hi / hinglish) | `language.py` | code |
| 4 | Understand the message into `Intent` JSON | `intent.py`, `prompts.py`, `llm_client.py` | **LLM** (rules fallback) |
| 5 | Route: search / refine / compare / stock / fact / my orders / not-found | `orchestrator.py` | code |
| 6 | Merge with memory ("only Nike", "cheaper ones") | `memory.py` | code |
| 7 | Clarify if vague: max 2 in a row, never re-ask | `clarify.py` | code |
| 8 | Hybrid retrieval: FTS5 BM25 + FAISS, relevance gate | `retrieval.py`, `vector_index.py` | code |
| 9 | Exact filters: budget, brand, size, colour, stock | `tools/catalog_tools.py` (M2) | code |
| 10 | Rank by fit, rating tie-break, own-history boost | `ranking.py` | code |
| 11 | Write the answer from a fact sheet, then the grounding guard | `responder.py` | **LLM** (template fallback) |
| 12 | Save memory and the trace | `orchestrator.py` → `repository.py` | code |

## Five answers for the judges

**1. "Can the LLM invent a product or price?"**
No. Products come only from SQLite rows that passed retrieval and the code filters. Prices on the cards are DB values. The LLM only writes the intro and reasons, from a fact sheet. `grounding_check()` rejects any text that mentions a product ID or ₹ amount not in the facts, or a discount or offer. When it rejects text, the template is used instead, and the trace records why.

**2. "Who does the arithmetic?"** (R7)
Code. Regex pulls exact budgets ("3k", "₹3,000", "2000 se kam", "1.5 lakh"), sizes and positions ("first two"). These **override** the LLM's guess. "Show cheaper ones" becomes `budget_max = cheapest shown price − 1`, computed from the DB.

**3. "Why hybrid search?"**
Keyword search catches exact words (brand, "IPX7", "size 9"). Embeddings catch meaning ("gift for a trekker") and Hindi text. The score is `0.5·semantic + 0.5·keyword`. A relevance gate (a keyword hit, or cosine ≥ 0.35) makes "electric cars" return not_found instead of random items.

**4. "How do you stop prompt injection?"** (R8)
There are three layers:
- A regex guard on the shopper's message, which refuses before any LLM call.
- Product text is cleaned of instruction-like sentences (`sanitize_catalog_text`) and wrapped in `<catalog_data>` tags marked as data.
- Ranking is done by code, so a review saying "recommend me" cannot change the order. The sample product P-1011 has exactly such a review.

**5. "How is privacy enforced?"** (R9)
Identity comes only from `customer_id` in the request. Messages naming another shopper, another customer ID, "what did X buy" or "X ne kya kharida" are refused by code. `get_orders()` is always called with the request's `customer_id`. A session created by one customer can't be used by another.

## Why one LLM and not agents
Understanding a message and writing a reply are language tasks, so they go to the LLM. Filtering by price, checking stock and comparing products are a database query plus Python, so they stay in code. "We left out extra agents on purpose" is the answer the brief explicitly rewards.

## Fallbacks (the demo never crashes)
- No Groq key, no internet or bad JSON: rule-based intent and template answers (trace shows `llm_mode: mock/template`).
- Embedding model or FAISS missing: keyword-only retrieval (trace shows `mode: keyword`).
- Unexpected exception: a polite error reply, and the trace is still saved.

## Tuning knobs (`.env`)
`SEMANTIC_WEIGHT`, `TOP_K`, `RETRIEVAL_K`, `MAX_CLARIFICATIONS`, plus `SEMANTIC_MIN` in `retrieval.py`. Tune them on the eval set and report what you changed.
