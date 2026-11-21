"""Phase 1 tests: config, contracts, schema and sample data are consistent."""
import json

import pytest
from pydantic import ValidationError

from app.config import LEGACY_CATEGORIES as CATEGORIES, settings  # sample fixture = legacy catalogue
from app.db.database import db_session, fts5_available, table_counts
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Constraints,
    Customer,
    Order,
    ProductIn,
    SessionState,
)
from tests.conftest import SAMPLE_DIR


def test_settings_defaults():
    assert settings.top_k >= 1
    assert settings.max_clarifications == 2
    assert settings.llm_mode.startswith(("mock", "groq:"))


def test_fts5_available():
    assert fts5_available(), "SQLite FTS5 is required for keyword search"


def test_sample_products_match_contract(sample_products):
    ids = [ProductIn(**p).product_id for p in sample_products]
    assert len(ids) == len(set(ids)), "product IDs must be unique"
    assert {p["category"] for p in sample_products} <= set(CATEGORIES)


def test_sample_customers_and_orders_valid():
    customers = [Customer(**c) for c in json.loads((SAMPLE_DIR / "customers.json").read_text("utf-8"))]
    orders = [Order(**o) for o in json.loads((SAMPLE_DIR / "orders.json").read_text("utf-8"))]
    product_ids = {p["product_id"] for p in json.loads((SAMPLE_DIR / "products.json").read_text("utf-8"))}
    customer_ids = {c.customer_id for c in customers}
    for order in orders:
        assert order.product_id in product_ids
        assert order.customer_id in customer_ids


def test_chat_request_validation():
    ok = ChatRequest(session_id="s1", customer_id="C-001", message="  running shoes  ")
    assert ok.message == "running shoes"
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", customer_id="priya", message="hi")      # bad id format
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", customer_id="C-001", message="   ")      # blank message


def test_official_response_example_parses():
    example = {
        "trace_id": "a41c9e07",
        "answer_type": "recommendation",
        "answer": "Here are two trekking gifts under Rs 3,000.",
        "products": [{"product_id": "P-2041", "name": "Trail Daypack 25L", "price": 2499,
                      "in_stock": True, "reason": "Light, under budget, rated 4.6 for trekking"}],
        "sources": [{"product_id": "P-2041", "fields": ["price", "stock", "description"]}],
        "follow_up_question": None,
    }
    parsed = ChatResponse(**example)
    assert parsed.products[0].price == 2499
    assert parsed.suggested_questions == []


def test_constraints_size_normalised():
    assert Constraints(size=9).size == "9"
    assert Constraints(size=9.0).size == "9"
    assert Constraints(size="M").size == "M"


def test_session_state_roundtrip():
    state = SessionState(customer_id="C-001", last_result_ids=["P-1004"])
    restored = SessionState.model_validate_json(state.model_dump_json())
    assert restored.last_result_ids == ["P-1004"]


def test_schema_and_sample_db(sample_db):
    counts = table_counts(sample_db)
    assert counts["products"] == 12
    assert counts["customers"] == 3
    assert counts["orders"] == 6


def test_fts_keyword_search(sample_db):
    with db_session(sample_db) as conn:
        rows = conn.execute(
            "SELECT product_id FROM products_fts WHERE products_fts MATCH ? ORDER BY bm25(products_fts)",
            ("speaker",),
        ).fetchall()
    found = {r["product_id"] for r in rows}
    assert {"P-1001", "P-1002", "P-1003"} <= found


def test_order_foreign_key_enforced(sample_db):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        with db_session(sample_db) as conn:
            conn.execute(
                "INSERT INTO orders (order_id, customer_id, product_id, order_date) VALUES (?,?,?,?)",
                ("O-99999", "C-001", "P-9999", "2026-11-21"),
            )
