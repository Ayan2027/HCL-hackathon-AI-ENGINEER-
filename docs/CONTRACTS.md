# Frozen contracts. Read before coding, change only with team agreement

Source of truth: `app/schemas.py`, `app/db/schema.sql`, `app/config.py`.

## IDs
| Entity | Format | Example |
|---|---|---|
| Product | `P-` + 4 digits | `P-1001` |
| Customer | `C-` + 3 digits | `C-001` |
| Order | `O-` + 5 digits | `O-00001` |
| Trace | 8 hex chars | `a41c9e07` |

## Categories (exact spelling)
Electronics · Footwear · Apparel · Sports & Outdoors · Home & Kitchen · Beauty & Personal Care

## Data files (Member 4 produces, Member 2 loads)
`data/raw/products.json` → list of `ProductIn`
`data/raw/customers.json` → list of `Customer`
`data/raw/orders.json` → list of `Order`
(`data/sample/` has the same format with 12 products for early testing.)

Product `attributes` keys used by the tools:
`color` (str), `sizes_available` (list of int or str), `material`, `capacity_l`, `battery_hours`, ...
Messy data is allowed: `brand`, `price`, `rating`, `weight_raw` may be null; `weight_raw` mixes `g`/`kg`/`ml`.

## HTTP API (Member 2 serves, Member 3 + 4 call)
| Method | Path | Body → Response |
|---|---|---|
| POST | `/chat` | `ChatRequest` → `ChatResponse` |
| GET | `/products/{id}` | → `ProductOut` or 404 |
| POST | `/ingest` | `IngestRequest` → `IngestResult` |
| GET | `/trace/{trace_id}` | → `TraceRecord` or 404 |
| GET | `/health` | → `HealthResponse` (200) |
| GET | `/customers` | → list of `CustomerSummary` (demo login selector only) |

## Python functions (Member 2 provides, Member 1 calls)
```python
# app/db/repository.py
get_product(pid) -> dict | None            get_products(ids) -> list[dict]
keyword_search(text, k=30) -> list[tuple[str, float]]     # (product_id, score; higher = better)
upsert_products(patches: list[ProductPatch]) -> IngestResult
get_customer(cid) -> dict | None           get_orders(cid) -> list[dict]
list_brands() -> list[str]                 list_categories() -> list[str]
get_session(sid) -> dict | None            save_session(sid, cid, state: SessionState) -> None
save_trace(record: TraceRecord) -> None    get_trace(tid) -> dict | None

# app/tools/catalog_tools.py
apply_filters(products, constraints) -> tuple[list[dict], list[dict]]   # (kept, dropped with reason)
sort_products(products, sort_by) -> list[dict]
check_stock(pid, size=None, color=None) -> StockResult
compare_products(ids) -> ComparisonTable
```
Product dicts returned by the repository have `attributes` and `reviews` already
parsed from JSON, plus `in_stock: bool` and `weight_g`.

## Python functions (Member 1 provides, Member 2 calls)
```python
# app/ai/orchestrator.py
handle_chat(request: ChatRequest) -> ChatResponse        # the ONLY call /chat makes
on_products_changed(product_ids: list[str]) -> bool      # /ingest calls this -> FAISS update
startup() -> None                                        # FastAPI startup: build/load index
vector_index_size() -> int                               # used by /health
```

## Golden rules
1. Prices, stock, filters, sorting and comparisons are computed in Python from SQLite. The LLM never computes them.
2. `customer_id` comes only from the request body. Text inside a message never changes identity.
3. Every `/chat` reply is saved to `audit_log` with a `trace_id`.
4. Product text (descriptions, reviews) is data, never instructions.
