# ShopSense AI: Intelligent Conversational Shopping Assistant

HCLTech Future Ready AI Engineer Hackathon · NIT Allahabad · 21 Nov 2026

A chat assistant that understands **English, Hindi and Hinglish**, searches **only our catalogue**, checks **price and stock with code**, explains every recommendation with a **product ID and a reason**, protects **shopper privacy**, ignores **injected instructions**, and saves a **replayable trace** for every reply.

> **Design in one line:** the LLM understands and explains; the code decides; the data stays true.

---

## Quick start

### Option A: Docker (what the judges run)
```bash
cp .env.example .env          # add GROQ_API_KEY, set MOCK_LLM=false for the real LLM
docker compose up --build
```
- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

Compose starts three services. **db** creates the SQLite schema and loads the data, then exits. **api** waits for it. **ui** waits for a healthy API.

### Option B: Local (development)
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env
python -m scripts.download_model          # embedding model, about 470 MB, once
python -m app.db.bootstrap --reset        # load data/raw into SQLite
uvicorn app.main:app --port 8000          # terminal 1
streamlit run ui/streamlit_app.py         # terminal 2
pytest                                    # 93 tests
```

### Rebuild the data and the evaluation
```bash
python -m scripts.generate_catalog        # 279 products (template generator, seed 42)
python -m scripts.generate_shoppers       # 30 shoppers + orders via Groq (--offline = seeded fallback)
python -m scripts.validate_data           # schema + every order points at a real product
python -m scripts.build_eval_set          # 50 questions built from the data
python -m app.db.bootstrap --reset
python -m scripts.run_eval                # needs the API running; or --inprocess
python -m scripts.export_audit_samples    # docs/AUDIT_SAMPLES.md from real traces
```

---

## Architecture

```
Streamlit UI ──HTTP──► FastAPI ──► orchestrator.handle_chat()
                                     1  session memory (bound to customer_id)        code
                                     2  privacy + injection guard                     code
                                     3  intent JSON (Groq LLM, regex fallback)        LLM
                                     4  route: search/refine/compare/stock/fact/...   code
                                     5  merge memory, clarify (max 2 in a row)        code
                                     6  hybrid retrieval: SQLite FTS5 + FAISS         code
                                     7  exact filters: budget, brand, size, stock     code
                                     8  product-type gate + rank by fit               code
                                     9  answer from fact sheet + grounding guard      LLM (template fallback)
                                    10  save memory + audit trace (trace_id)          code
                              SQLite: products · products_fts · customers · orders · sessions · audit_log
```

| Layer | Choice | Why |
|---|---|---|
| UI | Streamlit | Chat, cards and tables in pure Python |
| API | FastAPI | Typed contracts (Pydantic) and auto `/docs` |
| LLM | **Groq** `openai/gpt-oss-120b` (configurable) | See *Stack change* below |
| Retrieval | SQLite FTS5 (BM25) + FAISS (Sentence Transformers `paraphrase-multilingual-MiniLM-L12-v2`) | Keywords catch "Nike / size 9"; vectors catch meaning and Hindi |
| Storage | SQLite | One file; keyword index lives in the same DB, so it is always in sync |
| Tests | Pytest | 93 tests in mock mode, no network needed |

**Stack change (guide section 5):** the guide suggests a local model via Ollama. We use the hosted **Groq** API because it answers in about 1 s without a local GPU. `MOCK_LLM=true` keeps the whole system working offline, and the model is one `.env` setting. Groq retired `llama-3.3-70b-versatile` on 16 Aug 2026, and `openai/gpt-oss-120b` is its recommended replacement. `python -m scripts.check_setup` lists the models your key can use.

**Why one LLM and no agents:** understanding and explaining are language tasks, so they go to the LLM. Filtering, stock and comparison are a SQL query plus Python. More agents would add failure points without meeting any extra requirement.

---

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | `{session_id, customer_id, message}` → `{trace_id, answer_type, answer, products[], sources[], follow_up_question, suggested_questions, comparison, language}` |
| GET | `/products/{id}` | One product straight from SQLite (judges verify claims) |
| POST | `/ingest` | `{products:[{product_id, ...fields}]}`: add, or merge-update. The next answer reflects it |
| GET | `/trace/{trace_id}` | Replay: intent, retrieved candidates, filters with drop reasons, tools with timings, answer |
| GET | `/health` | 200 when ready; product count, vector count, LLM mode |
| GET | `/products`, `/customers` | Browse with the same exact filters; demo login list (ID, name, language only) |

`answer_type` is one of `recommendation · comparison · clarification · product_fact · not_found · refusal`.

---

## How each requirement is met

| Req | Implementation |
|---|---|
| R1 language | `language.py` detects en/hi/hinglish; Hindi and Hinglish words are mapped to English search terms; replies use the shopper's language |
| R2 clarify | `clarify.py`: buying-criteria questions, **max 2 in a row**, never re-asks a slot |
| R3 catalogue only | Products come only from SQLite rows; a relevance gate makes unrelated queries `not_found` |
| R4 reason + ID | Template reason built from DB facts; the LLM may reword it; the card carries the product ID |
| R5 memory | `SessionState` in SQLite: constraints, last results in order ("the first two"), clarification count |
| R6 no invention | Cards are built by code; `grounding_check()` rejects unknown IDs, ₹ amounts and "discounts" in LLM text |
| R7 code does the maths | `catalog_tools.py`: filters, stock, sort, compare. Regex budgets override the LLM's guess |
| R8 content is data | Message guard; review sanitising; `<catalog_data>` delimiters; ranking by code |
| R9 privacy | Identity only from `customer_id`; other-shopper questions refused; sessions bound to one shopper |
| R10 audit | `audit_log` row per reply, `GET /trace/{id}` |
| R11 live updates | `/ingest` updates the row, FTS5 and FAISS immediately |
| R12 evaluation | 50-question set built from the data; `run_eval.py` writes `docs/EVAL_REPORT.md` |

---

## Data
- **Catalogue:** 279 invented products in 6 categories, deliberately messy. Details in [docs/DATA_NOTE.md](docs/DATA_NOTE.md).
- **Shoppers:** 30 shoppers with 87 orders, generated by an LLM with the prompt saved verbatim ([scripts/prompts/shopper_prompt.md](scripts/prompts/shopper_prompt.md)). The model and raw replies are logged in `data/raw/generation_log.json`.
- **No real customer data** is used anywhere.

## Evaluation
See [docs/EVAL_REPORT.md](docs/EVAL_REPORT.md). It is generated by the script and lists the mode it ran in and every failure.

## Repository map
```
app/            config, schemas (contracts), main.py (API)
app/ai/         AI engine (Member 1)
app/db/         schema.sql, database, repository, bootstrap (Member 2)
app/tools/      exact catalogue tools (Member 2)
ui/             Streamlit app (Member 3)
scripts/        data generation, validation, eval, audit export, setup checks (Member 4)
data/raw/       generated catalogue, shoppers, orders, generation log
data/sample/    12-product dev dataset used by the tests
eval/           eval_set.json and results/
docs/           contracts, member guides, data note, eval report, audit samples, AI usage, contributions
tests/          93 pytest tests
```

## Team
See [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md) and [docs/AI_USAGE.md](docs/AI_USAGE.md).
