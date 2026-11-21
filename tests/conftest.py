"""
Shared pytest fixtures (Phase 1). Every member's tests can use:

    sample_db        -> path to a fresh temp SQLite DB loaded with data/sample/*
    sample_products  -> list of product dicts from data/sample/products.json

Tests ALWAYS run with MOCK_LLM=true and a temp database, so they never
touch the real DB or call Groq.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.config import settings
from app.db.database import db_session, init_db

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def _load(name: str) -> List[Dict[str, Any]]:
    return json.loads((SAMPLE_DIR / name).read_text(encoding="utf-8"))


def attributes_to_text(attributes: Dict[str, Any]) -> str:
    """{'color': 'Blue', 'sizes_available': [8, 9]} -> 'color Blue sizes available 8 9'"""
    parts = []
    for key, value in attributes.items():
        if isinstance(value, list):
            value = " ".join(str(v) for v in value)
        parts.append(f"{key.replace('_', ' ')} {value}")
    return " ".join(parts)


def insert_sample_data(db_path: Path) -> None:
    """Test-only loader using plain SQL. The real loader is repository.upsert_products() (Member 2)."""
    products = _load("products.json")
    customers = _load("customers.json")
    orders = _load("orders.json")
    with db_session(db_path) as conn:
        for p in products:
            conn.execute(
                """INSERT INTO products (product_id, name, category, subcategory, brand, price,
                       stock_qty, rating, rating_count, attributes_json, description,
                       review_summary, reviews_json, best_for, weight_raw, language)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (p["product_id"], p["name"], p["category"], p.get("subcategory"), p.get("brand"),
                 p.get("price"), p.get("stock_qty", 0), p.get("rating"), p.get("rating_count"),
                 json.dumps(p.get("attributes", {}), ensure_ascii=False), p.get("description"),
                 p.get("review_summary"), json.dumps(p.get("reviews", []), ensure_ascii=False),
                 p.get("best_for"), p.get("weight_raw"), p.get("language", "en")),
            )
            conn.execute(
                """INSERT INTO products_fts (product_id, name, brand, category, attributes_text, description)
                   VALUES (?,?,?,?,?,?)""",
                (p["product_id"], p["name"], p.get("brand") or "", p["category"],
                 attributes_to_text(p.get("attributes", {})), p.get("description") or ""),
            )
        for c in customers:
            conn.execute(
                """INSERT INTO customers (customer_id, name, city, preferred_language, budget_min,
                       budget_max, preferred_brands_json, preferred_categories_json, sizes_json, persona_note)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (c["customer_id"], c["name"], c.get("city"), c.get("preferred_language", "en"),
                 c.get("budget_min"), c.get("budget_max"), json.dumps(c.get("preferred_brands", [])),
                 json.dumps(c.get("preferred_categories", [])), json.dumps(c.get("sizes", {})),
                 c.get("persona_note")),
            )
        for o in orders:
            conn.execute(
                """INSERT INTO orders (order_id, customer_id, product_id, quantity, price_paid, order_date)
                   VALUES (?,?,?,?,?,?)""",
                (o["order_id"], o["customer_id"], o["product_id"], o.get("quantity", 1),
                 o.get("price_paid"), o["order_date"]),
            )


@pytest.fixture
def sample_products() -> List[Dict[str, Any]]:
    return _load("products.json")


@pytest.fixture
def sample_db(tmp_path, monkeypatch):
    """Fresh temp DB with sample data. Settings are pointed at it for the test only."""
    db_path = tmp_path / "test_shopsense.db"
    monkeypatch.setattr(settings, "db_path", db_path)
    monkeypatch.setattr(settings, "index_dir", tmp_path / "index")
    monkeypatch.setattr(settings, "mock_llm", True)
    monkeypatch.setattr(settings, "use_vector_search", False)   # fast tests, no model download
    init_db(db_path)
    insert_sample_data(db_path)
    return db_path
