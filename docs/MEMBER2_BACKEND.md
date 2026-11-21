# Member 2: backend and database

| File | What it does |
|---|---|
| `app/main.py` | FastAPI app: 5 official endpoints plus `GET /products` and `GET /customers`. Startup creates tables, auto-loads data if the DB is empty and builds the vector index |
| `app/db/repository.py` | **Every** SQL query lives here. Products, FTS5 keyword search, upsert for /ingest, customers, orders, sessions, audit log |
| `app/db/bootstrap.py` | Loads `data/raw/*.json` (or `data/sample/` if raw is empty), validating every record with Pydantic. Orders pointing at unknown products are skipped |
| `app/tools/normalize.py` | Messy data → exact values: `"0.95 kg"` → 950 g, size matching (9 vs "9", "m" vs "M") |
| `app/tools/catalog_tools.py` | Exact tools: `apply_filters`, `sort_products`, `check_stock`, `compare_products` |

## Run it

```bash
python -m app.db.bootstrap --reset            # (re)load data into SQLite
uvicorn app.main:app --reload --port 8000     # API
# open http://localhost:8000/docs  -> try every endpoint in the browser
pytest tests/test_backend.py -v
```

## Try it with curl (judges do this)

```bash
curl localhost:8000/health
curl localhost:8000/products/P-1007
curl -X POST localhost:8000/chat -H "content-type: application/json" \
  -d '{"session_id":"demo1","customer_id":"C-002","message":"trekking gift for my father under 3000"}'
curl localhost:8000/trace/<trace_id from the reply>
curl -X POST localhost:8000/ingest -H "content-type: application/json" \
  -d '{"products":[{"product_id":"P-1007","stock_qty":0}]}'      # next /chat no longer shows it
```
On Windows PowerShell, use `curl.exe` and escape the inner quotes, or just use `/docs`.

## Answers for the judges

**"Why are filters in code, not the LLM?"** (R7)
`apply_filters()` compares numbers exactly. A product with **no price never passes a budget filter**, and out-of-stock items are dropped with the reason `"out of stock"`. Every dropped product keeps its reason, so a not-found answer can say *why* (for example "2 over your budget").

**"How does /ingest work without a restart?"** (R11)
`upsert_products()` writes the row and its FTS5 keyword entry in the same transaction. Then `on_products_changed()` replaces the product's FAISS vector. Partial updates are merged: `{"product_id":"P-1007","stock_qty":0}` changes only stock.

**"Is keyword search safe from injection?"**
User text is split into word tokens, and each token is quoted (`"speaker"*`). FTS operators and SQL are treated as plain text, and all queries use `?` parameters.

**"How is shopper data isolated?"** (R9)
`get_orders(customer_id)` always filters by the ID from the request body. `/customers` returns only ID, name and language for the demo login picker. A session is tied to the customer who created it.

**"What's in the audit log?"** (R10)
One `audit_log` row per reply: the message, the intent (and whether it came from the LLM or rules), retrieved IDs with scores, filters and dropped reasons, each tool with its duration, final product IDs, the full response and latency. `GET /trace/{id}` returns it.

## Database design choices
- **SQLite + FTS5:** one file and zero setup. Keyword search is built into the same database, so it is always in sync.
- **A new short connection per query, plus WAL mode:** safe with FastAPI threads, and readers never block writers.
- **JSON columns** for `attributes` and `reviews`: each category has different attributes (sizes for shoes, battery for speakers) without needing 6 tables.
