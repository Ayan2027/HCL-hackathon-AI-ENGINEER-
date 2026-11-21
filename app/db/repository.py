"""
All SQL queries for ShopSense AI (Member 2).

Every other module talks to the database ONLY through these functions.
Product dicts returned here are "ready to use":
    attributes / reviews parsed from JSON, plus in_stock (bool) and weight_g.
"""
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.db.database import db_session
from app.schemas import (
    Customer,
    IngestResult,
    Order,
    ProductIn,
    ProductPatch,
    SessionState,
    TraceRecord,
)
from app.tools.normalize import attributes_to_text, clean_text, normalize_weight

_JSON_FIELDS = {"attributes_json": "attributes", "reviews_json": "reviews"}
_FTS_TOKEN_RE = re.compile(r"[0-9A-Za-zऀ-ॿ]+")
_FTS_STOPWORDS = {
    "a", "an", "the", "for", "and", "or", "of", "to", "in", "on", "with", "me", "my", "i",
    "is", "are", "show", "find", "need", "want", "some", "any", "please", "under", "below",
    "above", "best", "good", "one", "ones", "that", "this", "it", "can", "you", "do", "sell",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------
def _product_row_to_dict(row) -> Dict[str, Any]:
    product = dict(row)
    for column, key in _JSON_FIELDS.items():
        raw = product.pop(column, None)
        try:
            product[key] = json.loads(raw) if raw else ({} if key == "attributes" else [])
        except json.JSONDecodeError:
            product[key] = {} if key == "attributes" else []
    product["in_stock"] = (product.get("stock_qty") or 0) > 0
    return product


def get_product(product_id: str) -> Optional[Dict[str, Any]]:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM products WHERE product_id = ?", (product_id,)).fetchone()
    return _product_row_to_dict(row) if row else None


def get_products(product_ids: Iterable[str]) -> List[Dict[str, Any]]:
    """Fetch many products, keeping the order of product_ids. Unknown IDs are skipped."""
    ids = [pid for pid in dict.fromkeys(product_ids) if pid]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    with db_session() as conn:
        rows = conn.execute(f"SELECT * FROM products WHERE product_id IN ({placeholders})", ids).fetchall()
    by_id = {r["product_id"]: _product_row_to_dict(r) for r in rows}
    return [by_id[pid] for pid in ids if pid in by_id]


def get_all_products() -> List[Dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute("SELECT * FROM products ORDER BY product_id").fetchall()
    return [_product_row_to_dict(r) for r in rows]


def get_products_by_category(category: str, limit: int = 200) -> List[Dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute(
            "SELECT * FROM products WHERE lower(category) = lower(?) ORDER BY product_id LIMIT ?",
            (category, limit),
        ).fetchall()
    return [_product_row_to_dict(r) for r in rows]


def count_products() -> int:
    with db_session() as conn:
        return conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]


def list_brands() -> List[str]:
    with db_session() as conn:
        rows = conn.execute(
            "SELECT DISTINCT brand FROM products WHERE brand IS NOT NULL AND brand != '' ORDER BY brand"
        ).fetchall()
    return [r["brand"] for r in rows]


def list_categories() -> List[str]:
    with db_session() as conn:
        rows = conn.execute("SELECT DISTINCT category FROM products ORDER BY category").fetchall()
    return [r["category"] for r in rows]


def list_subcategories() -> List[str]:
    with db_session() as conn:
        rows = conn.execute(
            "SELECT DISTINCT subcategory FROM products WHERE subcategory IS NOT NULL ORDER BY subcategory"
        ).fetchall()
    return [r["subcategory"] for r in rows]


def _fts_query(text: str) -> str:
    """User text -> safe FTS5 query: '"running"* OR "shoes"*'. Quoting blocks FTS syntax injection."""
    tokens = [t.lower() for t in _FTS_TOKEN_RE.findall(text or "")]
    tokens = [t for t in dict.fromkeys(tokens) if t not in _FTS_STOPWORDS and len(t) > 1]
    return " OR ".join(f'"{t}"*' for t in tokens[:20])


def keyword_search(text: str, k: int = 30) -> List[Tuple[str, float]]:
    """BM25 keyword search. Returns [(product_id, score)] with higher score = better match."""
    query = _fts_query(text)
    if not query:
        return []
    with db_session() as conn:
        try:
            rows = conn.execute(
                """SELECT product_id, bm25(products_fts, 0.0, 10.0, 5.0, 3.0, 2.0, 1.0) AS rank
                   FROM products_fts WHERE products_fts MATCH ? ORDER BY rank LIMIT ?""",
                (query, k),
            ).fetchall()
        except Exception:  # noqa: BLE001 - a bad query must never crash a chat
            return []
    # bm25() is negative (more negative = better); flip the sign
    return [(r["product_id"], round(-float(r["rank"]), 4)) for r in rows]


def _write_product(conn, product: ProductIn) -> None:
    attributes_json = json.dumps(product.attributes, ensure_ascii=False)
    conn.execute(
        """INSERT INTO products (product_id, name, category, subcategory, brand, price, stock_qty,
               rating, rating_count, attributes_json, description, review_summary, reviews_json,
               best_for, weight_raw, weight_g, language, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(product_id) DO UPDATE SET
               name=excluded.name, category=excluded.category, subcategory=excluded.subcategory,
               brand=excluded.brand, price=excluded.price, stock_qty=excluded.stock_qty,
               rating=excluded.rating, rating_count=excluded.rating_count,
               attributes_json=excluded.attributes_json, description=excluded.description,
               review_summary=excluded.review_summary, reviews_json=excluded.reviews_json,
               best_for=excluded.best_for, weight_raw=excluded.weight_raw,
               weight_g=excluded.weight_g, language=excluded.language,
               updated_at=excluded.updated_at""",
        (product.product_id, product.name.strip(), product.category, clean_text(product.subcategory),
         clean_text(product.brand), product.price, product.stock_qty, product.rating,
         product.rating_count, attributes_json, clean_text(product.description),
         clean_text(product.review_summary), json.dumps(product.reviews, ensure_ascii=False),
         clean_text(product.best_for), clean_text(product.weight_raw),
         normalize_weight(product.weight_raw), product.language, _now()),
    )
    # Keep the keyword index in sync (delete + insert is simplest for FTS5)
    conn.execute("DELETE FROM products_fts WHERE product_id = ?", (product.product_id,))
    conn.execute(
        """INSERT INTO products_fts (product_id, name, brand, category, attributes_text, description)
           VALUES (?,?,?,?,?,?)""",
        (product.product_id, product.name,
         " ".join(filter(None, [product.brand, product.subcategory])) or "",
         product.category, attributes_to_text(product.attributes),
         " ".join(filter(None, [product.description, product.best_for, product.review_summary]))),
    )


def upsert_products(patches: List[ProductPatch]) -> IngestResult:
    """Insert new products or merge updates into existing ones (POST /ingest, R11).
    Changes are committed immediately, so the very next search sees them."""
    result = IngestResult()
    for patch in patches:
        existing = get_product(patch.product_id)
        changes = patch.model_dump(exclude_unset=True)
        if existing:
            merged = {**existing, **changes}
        else:
            if not changes.get("name") or not changes.get("category"):
                result.errors.append(f"{patch.product_id}: new products need name and category")
                continue
            merged = changes
        try:
            product = ProductIn(**merged)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"{patch.product_id}: {exc}")
            continue
        with db_session() as conn:
            _write_product(conn, product)
        if existing:
            result.updated += 1
        else:
            result.inserted += 1
        result.product_ids.append(product.product_id)
    return result


def insert_products(products: List[ProductIn]) -> int:
    """Bulk load full products (used by scripts/load_data.py)."""
    with db_session() as conn:
        for product in products:
            _write_product(conn, product)
    return len(products)


# ---------------------------------------------------------------------------
# Customers and orders (privacy: always filtered by customer_id)
# ---------------------------------------------------------------------------
def _customer_row_to_dict(row) -> Dict[str, Any]:
    customer = dict(row)
    customer["preferred_brands"] = json.loads(customer.pop("preferred_brands_json") or "[]")
    customer["preferred_categories"] = json.loads(customer.pop("preferred_categories_json") or "[]")
    customer["sizes"] = json.loads(customer.pop("sizes_json") or "{}")
    return customer


def get_customer(customer_id: str) -> Optional[Dict[str, Any]]:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
    return _customer_row_to_dict(row) if row else None


def list_customers() -> List[Dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute(
            "SELECT customer_id, name, preferred_language FROM customers ORDER BY customer_id"
        ).fetchall()
    return [dict(r) for r in rows]


def list_customer_names() -> List[Tuple[str, str]]:
    """[(customer_id, name)] - used ONLY by the privacy guard to spot questions about other shoppers."""
    with db_session() as conn:
        rows = conn.execute("SELECT customer_id, name FROM customers").fetchall()
    return [(r["customer_id"], r["name"]) for r in rows]


def get_orders(customer_id: str) -> List[Dict[str, Any]]:
    """Orders of ONE shopper, newest first, with product name/brand/category joined."""
    with db_session() as conn:
        rows = conn.execute(
            """SELECT o.order_id, o.customer_id, o.product_id, o.quantity, o.price_paid, o.order_date,
                      p.name AS product_name, p.brand, p.category
               FROM orders o JOIN products p ON p.product_id = o.product_id
               WHERE o.customer_id = ? ORDER BY o.order_date DESC""",
            (customer_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def insert_customers(customers: List[Customer]) -> int:
    with db_session() as conn:
        for c in customers:
            conn.execute(
                """INSERT OR REPLACE INTO customers (customer_id, name, city, preferred_language,
                       budget_min, budget_max, preferred_brands_json, preferred_categories_json,
                       sizes_json, persona_note) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (c.customer_id, c.name, c.city, c.preferred_language, c.budget_min, c.budget_max,
                 json.dumps(c.preferred_brands), json.dumps(c.preferred_categories),
                 json.dumps(c.sizes), c.persona_note),
            )
    return len(customers)


def insert_orders(orders: List[Order]) -> int:
    with db_session() as conn:
        for o in orders:
            conn.execute(
                """INSERT OR REPLACE INTO orders (order_id, customer_id, product_id, quantity,
                       price_paid, order_date) VALUES (?,?,?,?,?,?)""",
                (o.order_id, o.customer_id, o.product_id, o.quantity, o.price_paid, o.order_date),
            )
    return len(orders)


# ---------------------------------------------------------------------------
# Sessions (conversation memory)
# ---------------------------------------------------------------------------
def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Returns {'session_id', 'customer_id', 'state': dict} or None."""
    with db_session() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not row:
        return None
    data = dict(row)
    data["state"] = json.loads(data.pop("state_json") or "{}")
    return data


def save_session(session_id: str, customer_id: str, state: SessionState) -> None:
    with db_session() as conn:
        conn.execute(
            """INSERT INTO sessions (session_id, customer_id, state_json, updated_at)
               VALUES (?,?,?,?)
               ON CONFLICT(session_id) DO UPDATE SET state_json=excluded.state_json,
                   updated_at=excluded.updated_at""",
            (session_id, customer_id, state.model_dump_json(), _now()),
        )


# ---------------------------------------------------------------------------
# Audit log (R10)
# ---------------------------------------------------------------------------
def save_trace(record: TraceRecord) -> None:
    def dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    with db_session() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO audit_log (trace_id, session_id, customer_id, created_at, message,
                   language, intent_json, retrieved_json, filters_json, tools_json,
                   final_product_ids_json, answer_type, answer, follow_up_question, response_json,
                   latency_ms, llm_mode)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (record.trace_id, record.session_id, record.customer_id, record.created_at,
             record.message, record.language, dumps(record.intent), dumps(record.retrieved),
             dumps(record.filters), dumps([t.model_dump() for t in record.tools]),
             dumps(record.final_product_ids), record.answer_type, record.answer,
             record.follow_up_question, dumps(record.response), record.latency_ms, record.llm_mode),
        )


def get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM audit_log WHERE trace_id = ?", (trace_id,)).fetchone()
    if not row:
        return None
    data = dict(row)
    mapping = {
        "intent_json": "intent", "retrieved_json": "retrieved", "filters_json": "filters",
        "tools_json": "tools", "final_product_ids_json": "final_product_ids", "response_json": "response",
    }
    for column, key in mapping.items():
        data[key] = json.loads(data.pop(column) or "null")
    return data
