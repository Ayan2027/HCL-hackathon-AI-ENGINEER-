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

_JSON_FIELDS = {
    "attributes_json": ("attributes", dict),
    "reviews_json": ("reviews", list),
    "quality_flags_json": ("quality_flags", list),
    "original_values_json": ("original_values", dict),
}
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
    for column, (key, empty) in _JSON_FIELDS.items():
        raw = product.pop(column, None)
        try:
            product[key] = json.loads(raw) if raw else empty()
        except json.JSONDecodeError:
            product[key] = empty()
    product["in_stock"] = (product.get("stock_qty") or 0) > 0
    product["stock_is_simulated"] = bool(product.get("stock_is_simulated") or 0)
    product["is_active"] = bool(product.get("is_active", 1))
    return product


_ACTIVE = " AND is_active = 1"


def get_product(product_id: str, include_inactive: bool = False) -> Optional[Dict[str, Any]]:
    """One product. Archived products (legacy order history) are hidden unless asked for."""
    sql = "SELECT * FROM products WHERE product_id = ?" + ("" if include_inactive else _ACTIVE)
    with db_session() as conn:
        row = conn.execute(sql, (product_id,)).fetchone()
    return _product_row_to_dict(row) if row else None


def get_products(product_ids: Iterable[str], include_inactive: bool = False) -> List[Dict[str, Any]]:
    """Fetch many products, keeping the order of product_ids. Unknown/archived IDs are skipped."""
    ids = [pid for pid in dict.fromkeys(product_ids) if pid]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    sql = f"SELECT * FROM products WHERE product_id IN ({placeholders})" + ("" if include_inactive else _ACTIVE)
    with db_session() as conn:
        rows = conn.execute(sql, ids).fetchall()
    by_id = {r["product_id"]: _product_row_to_dict(r) for r in rows}
    return [by_id[pid] for pid in ids if pid in by_id]


def get_all_products(include_inactive: bool = False) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM products" + ("" if include_inactive else " WHERE is_active = 1") + " ORDER BY product_id"
    with db_session() as conn:
        rows = conn.execute(sql).fetchall()
    return [_product_row_to_dict(r) for r in rows]


def get_products_by_category(category: str, limit: int = 200) -> List[Dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute(
            "SELECT * FROM products WHERE lower(category) = lower(?) AND is_active = 1 "
            "ORDER BY product_id LIMIT ?",
            (category, limit),
        ).fetchall()
    return [_product_row_to_dict(r) for r in rows]


def count_products() -> int:
    """Products in the current (active) catalogue."""
    with db_session() as conn:
        return conn.execute("SELECT COUNT(*) FROM products WHERE is_active = 1").fetchone()[0]


def count_archived_products() -> int:
    with db_session() as conn:
        return conn.execute("SELECT COUNT(*) FROM products WHERE is_active = 0").fetchone()[0]


def any_simulated_stock() -> bool:
    with db_session() as conn:
        return bool(conn.execute(
            "SELECT 1 FROM products WHERE is_active = 1 AND stock_is_simulated = 1 LIMIT 1").fetchone())


def list_brands() -> List[str]:
    with db_session() as conn:
        rows = conn.execute(
            "SELECT DISTINCT brand FROM products WHERE brand IS NOT NULL AND brand != '' AND is_active = 1 "
            "ORDER BY brand"
        ).fetchall()
    return [r["brand"] for r in rows]


def list_categories() -> List[str]:
    with db_session() as conn:
        rows = conn.execute(
            "SELECT DISTINCT category FROM products WHERE is_active = 1 ORDER BY category").fetchall()
    return [r["category"] for r in rows]


_CONNECTORS = {"for", "with", "of", "and", "in", "by", "set", "pack", "combo", "to"}


def catalog_terms() -> set:
    """Product-type words (head nouns) of the active catalogue, for the 'do we sell this?' check.
    A name word counts only when it is not modifying the next word: 'Water Bottle 1L' -> bottle,
    but 'Shoe Rack' does NOT make 'shoe' a product we sell. The last word of every subcategory
    leaf ('Shoe Racks' -> racks) counts as well."""
    with db_session() as conn:
        rows = conn.execute("SELECT name, subcategory FROM products WHERE is_active = 1").fetchall()
    words = set()
    for r in rows:
        tokens = [w.lower() for w in _FTS_TOKEN_RE.findall(r["name"] or "")]
        for i, token in enumerate(tokens):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is None or not nxt.isalpha() or nxt in _CONNECTORS:
                words.add(token)
        leaf = (r["subcategory"] or "").split(">")[-1]
        leaf_tokens = [w.lower() for w in _FTS_TOKEN_RE.findall(leaf)]
        if leaf_tokens:
            words.add(leaf_tokens[-1])
    return words


def list_subcategories() -> List[str]:
    with db_session() as conn:
        rows = conn.execute(
            "SELECT DISTINCT subcategory FROM products WHERE subcategory IS NOT NULL AND is_active = 1 "
            "ORDER BY subcategory"
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


_PRODUCT_COLUMNS = [
    "product_id", "name", "category", "subcategory", "brand", "price", "stock_qty", "rating",
    "rating_count", "attributes_json", "description", "review_summary", "reviews_json", "best_for",
    "weight_raw", "weight_g", "language", "retail_price", "image_url", "source_product_id",
    "data_source", "stock_is_simulated", "quality_flags_json", "original_values_json",
    "embedding_text", "is_active",
]


def product_db_values(product: ProductIn) -> Dict[str, Any]:
    """The exact column values written for a product (also used to detect unchanged rows)."""
    return {
        "product_id": product.product_id,
        "name": product.name.strip(),
        "category": product.category,
        "subcategory": clean_text(product.subcategory),
        "brand": clean_text(product.brand),
        "price": product.price,
        "stock_qty": product.stock_qty,
        "rating": product.rating,
        "rating_count": product.rating_count,
        "attributes_json": json.dumps(product.attributes, ensure_ascii=False),
        "description": clean_text(product.description),
        "review_summary": clean_text(product.review_summary),
        "reviews_json": json.dumps(product.reviews, ensure_ascii=False),
        "best_for": clean_text(product.best_for),
        "weight_raw": clean_text(product.weight_raw),
        "weight_g": normalize_weight(product.weight_raw),
        "language": product.language,
        "retail_price": product.retail_price,
        "image_url": clean_text(product.image_url),
        "source_product_id": clean_text(product.source_product_id),
        "data_source": clean_text(product.data_source),
        "stock_is_simulated": 1 if product.stock_is_simulated else 0,
        "quality_flags_json": json.dumps(product.quality_flags, ensure_ascii=False),
        "original_values_json": json.dumps(product.original_values, ensure_ascii=False),
        "embedding_text": clean_text(product.embedding_text),
        "is_active": 1 if product.is_active else 0,
    }


# Boilerplate / identifier attributes kept out of keyword search: "covered_in_warranty" would
# otherwise make every product match "cover"
FTS_SKIP_ATTRIBUTES = {"covered_in_warranty", "not_covered_in_warranty", "warranty_service_type",
                       "sales_package", "additional_info", "model_id", "style_code", "design_code",
                       "number_of_contents_in_sales_package"}


def _fts_row(product: Dict[str, Any]) -> Tuple:
    """Searchable text for one product: name, brand + subcategory, category, attributes, description.
    Price, stock and rating are NOT indexed - they are filtered exactly in code."""
    attributes = product.get("attributes")
    if attributes is None:
        try:
            attributes = json.loads(product.get("attributes_json") or "{}")
        except json.JSONDecodeError:
            attributes = {}
    attributes = {k: v for k, v in attributes.items() if k not in FTS_SKIP_ATTRIBUTES}
    subcategory = (product.get("subcategory") or "").replace(">", " ")
    return (
        product["product_id"], product["name"],
        " ".join(filter(None, [product.get("brand"), subcategory])) or "",
        product["category"], attributes_to_text(attributes),
        " ".join(filter(None, [product.get("description"), product.get("best_for"), product.get("review_summary")])),
    )


def _write_fts(conn, product: Dict[str, Any]) -> None:
    conn.execute("DELETE FROM products_fts WHERE product_id = ?", (product["product_id"],))
    if product.get("is_active", 1):
        conn.execute(
            """INSERT INTO products_fts (product_id, name, brand, category, attributes_text, description)
               VALUES (?,?,?,?,?,?)""",
            _fts_row(product),
        )


def _write_product(conn, product: ProductIn) -> None:
    values = product_db_values(product)
    columns = ", ".join(_PRODUCT_COLUMNS + ["updated_at"])
    placeholders = ",".join("?" for _ in range(len(_PRODUCT_COLUMNS) + 1))
    updates = ", ".join(f"{c}=excluded.{c}" for c in _PRODUCT_COLUMNS[1:] + ["updated_at"])
    conn.execute(
        f"INSERT INTO products ({columns}) VALUES ({placeholders}) "
        f"ON CONFLICT(product_id) DO UPDATE SET {updates}",
        [values[c] for c in _PRODUCT_COLUMNS] + [_now()],
    )
    # Keep the keyword index in sync (delete + insert is simplest for FTS5); archived rows are not indexed
    _write_fts(conn, {**values, "attributes": product.attributes})


def rebuild_fts(conn) -> int:
    """Rebuild the whole keyword index from the active catalogue. Returns rows indexed."""
    conn.execute("DELETE FROM products_fts")
    rows = conn.execute("SELECT * FROM products WHERE is_active = 1").fetchall()
    for row in rows:
        _write_fts(conn, dict(row))
    return len(rows)


def upsert_products(patches: List[ProductPatch]) -> IngestResult:
    """Insert new products or merge updates into existing ones (POST /ingest, R11).
    Changes are committed immediately, so the very next search sees them."""
    result = IngestResult()
    for patch in patches:
        existing = get_product(patch.product_id, include_inactive=True)
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
    """Orders of ONE shopper, newest first, with product name/brand/category joined.
    product_active is 0 for archived legacy products (kept only as a historical record)
    and NULL if the product row is missing entirely."""
    with db_session() as conn:
        rows = conn.execute(
            """SELECT o.order_id, o.customer_id, o.product_id, o.quantity, o.price_paid, o.order_date,
                      p.name AS product_name, p.brand, p.category, p.is_active AS product_active
               FROM orders o LEFT JOIN products p ON p.product_id = o.product_id
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
