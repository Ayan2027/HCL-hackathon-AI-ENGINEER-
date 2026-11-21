"""
Load JSON data into SQLite (Member 2).

    python -m app.db.bootstrap            # load data/raw (or data/sample) if the DB is empty
    python -m app.db.bootstrap --reset    # wipe the DB and reload
    python -m app.db.bootstrap --if-empty # used by docker compose "db" service

Source folder: data/raw/ when products.json exists there (Member 4's full
dataset), otherwise data/sample/ (12 products for early development).
The API also calls load_if_empty() at startup, so `docker compose up`
works on a fresh machine with no manual steps.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

from app.config import settings
from app.db import repository as repo
from app.db.database import init_db, reset_db, table_counts
from app.schemas import Customer, Order, ProductIn

logger = logging.getLogger("shopsense.bootstrap")


def data_source_dir() -> Path:
    if (settings.raw_dir / "products.json").exists():
        return settings.raw_dir
    return settings.sample_dir


def _read(folder: Path, name: str) -> list:
    path = folder / name
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_all(folder: Path = None) -> Dict[str, int]:
    """Validate every record with the shared Pydantic models, then insert.
    Orders that point at unknown products/customers are skipped and reported."""
    folder = folder or data_source_dir()
    products = [ProductIn(**p) for p in _read(folder, "products.json")]
    customers = [Customer(**c) for c in _read(folder, "customers.json")]
    product_ids = {p.product_id for p in products}
    customer_ids = {c.customer_id for c in customers}

    orders, skipped = [], 0
    for raw in _read(folder, "orders.json"):
        order = Order(**raw)
        if order.product_id in product_ids and order.customer_id in customer_ids:
            orders.append(order)
        else:
            skipped += 1
            logger.warning("Skipping order %s: unknown product/customer", order.order_id)

    repo.insert_products(products)
    repo.insert_customers(customers)
    repo.insert_orders(orders)
    return {"source": str(folder), "products": len(products), "customers": len(customers),
            "orders": len(orders), "orders_skipped": skipped}


def load_if_empty() -> Dict[str, int]:
    init_db()
    if table_counts()["products"] > 0:
        return {"skipped": 1}
    return load_all()


def main() -> int:
    parser = argparse.ArgumentParser(description="Load catalogue, shoppers and orders into SQLite")
    parser.add_argument("--reset", action="store_true", help="delete the database first")
    parser.add_argument("--source", type=Path, help="folder with products.json, customers.json, orders.json")
    parser.add_argument("--if-empty", action="store_true", help="load only when the products table is empty")
    args = parser.parse_args()

    if args.if_empty:
        print(json.dumps(load_if_empty()), "| counts:", table_counts())
        return 0
    if args.reset:
        reset_db()
    else:
        init_db()
    result = load_all(args.source)
    print(json.dumps(result, indent=2))
    print("Table counts:", table_counts())
    print("Tip: the FAISS index rebuilds automatically when the API starts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
