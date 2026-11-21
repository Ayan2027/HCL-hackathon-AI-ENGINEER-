"""Member 4 tests: data generators, validator and eval set meet the brief."""
import json
from pathlib import Path

from scripts import generate_catalog, generate_shoppers
from scripts.validate_data import validate


def _write_dataset(folder: Path) -> None:
    result = generate_catalog.generate(seed=42)
    products = result["products"]
    oos = [p["product_id"] for p in products if p["stock_qty"] == 0]
    shoppers = generate_shoppers.generate_fallback(products, oos, seed=42)
    customers, orders, _ = generate_shoppers.clean(shoppers, products, oos, seed=42)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "products.json").write_text(json.dumps(products, ensure_ascii=False), encoding="utf-8")
    (folder / "customers.json").write_text(json.dumps(customers), encoding="utf-8")
    (folder / "orders.json").write_text(json.dumps(orders), encoding="utf-8")


def test_catalogue_is_reproducible():
    a = generate_catalog.generate(seed=42)["products"]
    b = generate_catalog.generate(seed=42)["products"]
    assert a == b


def test_catalogue_has_required_mess():
    meta = generate_catalog.generate(seed=42)["meta"]
    for key in ["out_of_stock", "low_stock", "missing_brand", "missing_rating", "hindi_description",
                "injection_reviews", "twins"]:
        assert meta[key], f"no {key} defects generated"


def test_full_dataset_passes_validation(tmp_path):
    _write_dataset(tmp_path)
    results = validate(tmp_path)
    failed = [label for ok, label in results if not ok]
    assert not failed, failed


def test_cleanup_drops_bad_llm_output():
    products = generate_catalog.generate(seed=42)["products"]
    oos = [p["product_id"] for p in products if p["stock_qty"] == 0]
    messy = [{"customer_id": "C-001", "name": "X", "preferred_language": "klingon",
              "preferred_brands": ["NotABrand"], "preferred_categories": ["Toys"],
              "orders": [{"product_id": "P-9999"}, {"product_id": products[0]["product_id"], "quantity": 9}]}]
    customers, orders, report = generate_shoppers.clean(messy, products, oos, seed=1)
    c1 = customers[0]
    assert c1["name"] == "Priya Sharma" and c1["preferred_language"] == "en"
    assert c1["preferred_brands"] == [] and c1["preferred_categories"][0] != "Toys"
    assert report["dropped_invalid_order_refs"] >= 1
    valid_ids = {p["product_id"] for p in products}
    assert all(o["product_id"] in valid_ids for o in orders)
    assert all(1 <= o["quantity"] <= 3 for o in orders)


def test_eval_set_meets_minimums():
    items = json.loads((Path(__file__).resolve().parent.parent / "eval" / "eval_set.json").read_text("utf-8"))
    counts = {}
    for item in items:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    assert len(items) >= 40
    minimums = {"vague": 6, "not_found": 4, "constraint": 5, "follow_up": 4, "multilingual": 3, "privacy": 3, "injection": 2}
    for category, minimum in minimums.items():
        assert counts.get(category, 0) >= minimum, (category, counts.get(category))
