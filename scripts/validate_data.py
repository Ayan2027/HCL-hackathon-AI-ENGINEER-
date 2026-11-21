"""
Validate the dataset against the schema and the brief's minimums (Member 4).

    python -m scripts.validate_data            # checks data/raw
    python -m scripts.validate_data --dir data/sample --relaxed

Exit code 0 = all checks passed, 1 = something failed (printed with details).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, List, Tuple

from pydantic import ValidationError

from app.config import BASE_DIR, CATEGORIES
from app.schemas import Customer, Order, ProductIn
from app.tools.normalize import normalize_weight


def validate(folder: Path, relaxed: bool = False) -> List[Tuple[bool, str]]:
    results: List[Tuple[bool, str]] = []

    def check(ok: bool, label: str) -> None:
        results.append((bool(ok), label))

    def load(name: str) -> list:
        return json.loads((folder / name).read_text(encoding="utf-8"))

    raw_products, raw_customers, raw_orders = load("products.json"), load("customers.json"), load("orders.json")

    def parse(model: Callable, rows: list, kind: str) -> list:
        parsed, errors = [], []
        for row in rows:
            try:
                parsed.append(model(**row))
            except ValidationError as exc:
                errors.append(f"{row.get('product_id') or row.get('customer_id') or row.get('order_id')}: "
                              f"{exc.errors()[0]['msg']}")
        check(not errors, f"{kind}: all {len(rows)} records match the schema" + (f" ({errors[:3]})" if errors else ""))
        return parsed

    products = parse(ProductIn, raw_products, "products")
    customers = parse(Customer, raw_customers, "customers")
    orders = parse(Order, raw_orders, "orders")

    pids = [p.product_id for p in products]
    cids = [c.customer_id for c in customers]
    check(len(pids) == len(set(pids)), "product IDs are unique")
    check(len(cids) == len(set(cids)), "customer IDs are unique")
    check(len({o.order_id for o in orders}) == len(orders), "order IDs are unique")
    bad_refs = [o.order_id for o in orders if o.product_id not in set(pids)]
    check(not bad_refs, f"every order references a valid product ({len(bad_refs)} bad: {bad_refs[:5]})")
    bad_cust = [o.order_id for o in orders if o.customer_id not in set(cids)]
    check(not bad_cust, f"every order references a valid customer ({len(bad_cust)} bad)")
    unknown_cat = [p.product_id for p in products if p.category not in CATEGORIES]
    check(not unknown_cat, f"categories are from the fixed list ({unknown_cat[:5]})")

    orders_per = Counter(o.customer_id for o in orders)
    with_3 = sum(1 for c in cids if orders_per[c] >= 3)
    if not relaxed:
        cats = Counter(p.category for p in products)
        check(len(products) >= 250, f"at least 250 products (have {len(products)})")
        check(len(cats) >= 5, f"at least 5 categories (have {len(cats)}: {dict(cats)})")
        check(len(customers) >= 25, f"at least 25 shoppers (have {len(customers)})")
        check(with_3 / max(1, len(customers)) > 0.5, f"most shoppers have 3+ orders ({with_3}/{len(customers)})")
        check(any(orders_per[c] == 0 for c in cids), "edge case: a brand-new shopper with no orders")
        oos = {p.product_id for p in products if p.stock_qty == 0}
        check(any(sum(1 for o in orders if o.customer_id == c and o.product_id in oos) >= 2 for c in cids),
              "edge case: a shopper whose favourites are out of stock")
        check(len(oos) > 0 and any(0 < p.stock_qty <= 3 for p in products), "out-of-stock and nearly-out products exist")
        check(any(p.brand is None for p in products) and any(p.rating is None for p in products),
              "some products have missing fields (brand / rating)")
        units = {str(p.weight_raw).split()[-1].lower().lstrip("0123456789.") for p in products if p.weight_raw}
        check({"g", "kg"} <= units, f"weights use mixed units {sorted(units)}")
        unparsable = [p.product_id for p in products if p.weight_raw and normalize_weight(p.weight_raw) is None]
        check(not unparsable, f"every weight can be normalised to grams ({unparsable[:5]})")
        check(any(p.language == "hi" for p in products), "some descriptions are in Hindi")
        langs = Counter(c.preferred_language for c in customers)
        check(len(langs) >= 2, f"shoppers prefer different languages {dict(langs)}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=BASE_DIR / "data" / "raw")
    parser.add_argument("--relaxed", action="store_true", help="schema + reference checks only")
    args = parser.parse_args()
    results = validate(args.dir, args.relaxed)
    for ok, label in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    failed = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
