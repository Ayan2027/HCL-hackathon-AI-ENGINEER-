"""
Member 2 tests: normalisation, exact tools, repository and the 5 official endpoints.
Run: pytest tests/test_backend.py -v
"""
import pytest
from fastapi.testclient import TestClient

from app.db import repository as repo
from app.schemas import Constraints, ProductPatch
from app.tools.catalog_tools import apply_filters, check_stock, compare_products, sort_products
from app.tools.normalize import normalize_weight, size_matches


# ---------------------------------------------------------------- normalisation (messy units)
@pytest.mark.parametrize("raw,grams", [
    ("350 g", 350.0), ("0.95 kg", 950.0), ("200g", 200.0), ("0.5kg", 500.0),
    ("100 ml", None), (None, None), ("heavy", None),
])
def test_normalize_weight(raw, grams):
    assert normalize_weight(raw) == grams


def test_size_matches_numbers_and_letters():
    assert size_matches("9", [7, 8, 9]) is True
    assert size_matches("m", ["S", "M", "L"]) is True
    assert size_matches("11", [7, 8, 9]) is False
    assert size_matches("9", None) is None


def test_weight_normalised_on_load(sample_db):
    repo.upsert_products([ProductPatch(product_id="P-1002", weight_raw="0.95 kg")])
    assert repo.get_product("P-1002")["weight_g"] == 950.0


# ---------------------------------------------------------------- exact filters (R7)
def test_apply_filters_budget_brand_stock(sample_db):
    products = repo.get_all_products()
    kept, dropped = apply_filters(products, Constraints(category="Electronics", budget_max=1500))
    assert {p["product_id"] for p in kept} == {"P-1001", "P-1003"}
    reasons = {d["product_id"]: d["reason"] for d in dropped}
    assert reasons["P-1002"] == "over budget"

    kept, dropped = apply_filters(products, Constraints(brands=["jbl"]))
    assert kept == []                                         # JBL speaker is out of stock
    assert {"product_id": "P-1002", "reason": "out of stock"} in dropped


def test_missing_price_never_passes_budget(sample_db):
    repo.upsert_products([ProductPatch(product_id="P-1090", name="Mystery Item", category="Apparel")])
    kept, dropped = apply_filters([repo.get_product("P-1090")], Constraints(budget_max=10_000))
    assert not kept and dropped[0]["reason"] == "price not listed"


def test_size_and_colour_filters(sample_db):
    shoes = [p for p in repo.get_all_products() if p["category"] == "Footwear"]
    kept, _ = apply_filters(shoes, Constraints(size="9", color="blue"))
    assert [p["product_id"] for p in kept] == ["P-1004"]


def test_sort_products_missing_values_last(sample_db):
    products = repo.get_products(["P-1001", "P-1003", "P-1002"])
    assert [p["product_id"] for p in sort_products(products, "price_asc")] == ["P-1003", "P-1001", "P-1002"]
    assert sort_products(products, "rating")[-1]["product_id"] == "P-1003"      # rating missing


def test_check_stock(sample_db):
    assert check_stock("P-1004", size="9").size_available is True
    r = check_stock("P-1004", size="11")
    assert r.size_available is False and not r.in_stock
    assert check_stock("P-1002").in_stock is False
    assert check_stock("P-9999").found is False


def test_compare_products_marks_missing(sample_db):
    table = compare_products(["P-1001", "P-1003"])
    assert [r.product_id for r in table.rows] == ["P-1001", "P-1003"]
    assert table.rows[1].values["rating"] == "Not available in catalogue"   # missing, never shown as 0
    assert "price" in table.fields


# ---------------------------------------------------------------- repository
def test_keyword_search_is_injection_safe(sample_db):
    assert repo.keyword_search('speaker" OR 1=1; DROP TABLE products; --')  # no crash, still works
    assert repo.count_products() == 12
    assert repo.keyword_search("") == []


def test_upsert_merges_partial_update(sample_db):
    result = repo.upsert_products([ProductPatch(product_id="P-1001", price=1199)])
    assert result.updated == 1
    p = repo.get_product("P-1001")
    assert p["price"] == 1199 and p["brand"] == "boAt"          # other fields kept


def test_new_product_needs_name_and_category(sample_db):
    result = repo.upsert_products([ProductPatch(product_id="P-1095", price=10)])
    assert result.inserted == 0 and result.errors


def test_orders_are_per_customer(sample_db):
    assert {o["customer_id"] for o in repo.get_orders("C-001")} == {"C-001"}
    assert repo.get_orders("C-003") == []


# ---------------------------------------------------------------- API endpoints
@pytest.fixture
def client(sample_db):
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["products"] == 12


def test_get_product_and_404(client):
    assert client.get("/products/P-1007").json()["price"] == 2499
    assert client.get("/products/p-1007").status_code == 200
    assert client.get("/products/P-9999").status_code == 404


def test_chat_contract_and_trace(client):
    r = client.post("/chat", json={"session_id": "api-1", "customer_id": "C-002",
                                   "message": "trekking backpack under 3000"})
    assert r.status_code == 200
    body = r.json()
    for key in ("trace_id", "answer_type", "answer", "products", "sources", "follow_up_question"):
        assert key in body
    assert body["answer_type"] == "recommendation"
    trace = client.get(f"/trace/{body['trace_id']}").json()
    assert trace["message"] == "trekking backpack under 3000"
    assert client.get("/trace/nope0000").status_code == 404


def test_chat_validation(client):
    assert client.post("/chat", json={"session_id": "x", "customer_id": "Priya", "message": "hi"}).status_code == 422
    assert client.post("/chat", json={"session_id": "x", "customer_id": "C-001", "message": ""}).status_code == 422


def test_ingest_then_chat_sees_change(client):
    r = client.post("/ingest", json={"products": [{
        "product_id": "P-1098", "name": "AquaGuard Steel Water Bottle 1L", "category": "Sports & Outdoors",
        "brand": "Milton", "price": 699, "stock_qty": 10, "description": "Insulated steel bottle for gym and treks"}]})
    assert r.status_code == 200 and r.json()["inserted"] == 1
    chat = client.post("/chat", json={"session_id": "api-2", "customer_id": "C-001",
                                      "message": "steel water bottle"}).json()
    assert "P-1098" in [p["product_id"] for p in chat["products"]]

    client.post("/ingest", json={"products": [{"product_id": "P-1098", "price": 1499}]})
    assert client.get("/products/P-1098").json()["price"] == 1499


def test_ingest_rejects_bad_new_product(client):
    r = client.post("/ingest", json={"products": [{"product_id": "P-1097", "price": 5}]})
    assert r.status_code == 422


def test_products_list_filters(client):
    rows = client.get("/products", params={"category": "Electronics", "max_price": 1500, "in_stock": True,
                                           "sort_by": "price_asc"}).json()
    assert [r["product_id"] for r in rows] == ["P-1003", "P-1001"]


def test_customers_list_exposes_no_orders(client):
    rows = client.get("/customers").json()
    assert len(rows) == 3
    assert set(rows[0].keys()) == {"customer_id", "name", "preferred_language"}
