"""
Deterministic catalogue tools (Member 2) - R7: "Let code do the exact work".

Price filters, brand/size/colour checks, stock checks, sorting and comparison
tables are computed here from database values. The LLM only explains results.
"""
from typing import Any, Dict, List, Optional, Tuple

from app.db import repository as repo
from app.schemas import ComparisonRow, ComparisonTable, Constraints, SortBy, StockResult
from app.tools.normalize import color_matches, size_matches

NOT_AVAILABLE = "Not available in catalogue"
SIMULATED_NOTE = "simulated demo inventory"


def stock_note(product: Dict[str, Any]) -> str:
    """' (simulated demo inventory)' for demo stock, so it is never presented as live retailer stock."""
    return f" ({SIMULATED_NOTE})" if product.get("stock_is_simulated") else ""


def _lower(values: List[str]) -> List[str]:
    return [v.strip().lower() for v in values if v and v.strip()]


def apply_filters(
    products: List[Dict[str, Any]], constraints: Constraints
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Exact hard filters. Returns (kept, dropped) where dropped items carry a 'reason'.
    Order of kept products is preserved (ranking happens elsewhere)."""
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    want_brands = _lower(constraints.brands)
    block_brands = _lower(constraints.exclude_brands)

    for p in products:
        reason: Optional[str] = None
        price = p.get("price")
        brand = (p.get("brand") or "").strip().lower()
        attrs = p.get("attributes") or {}

        if constraints.category and (p.get("category") or "").lower() != constraints.category.lower():
            reason = "different category"
        elif constraints.budget_max is not None and price is None:
            reason = "price not listed"
        elif constraints.budget_max is not None and price > constraints.budget_max:
            reason = "over budget"
        elif constraints.budget_min is not None and (price is None or price < constraints.budget_min):
            reason = "below minimum price"
        elif want_brands and brand not in want_brands:
            reason = "different brand"
        elif block_brands and brand in block_brands:
            reason = "excluded brand"
        elif constraints.size is not None and size_matches(constraints.size, attrs.get("sizes_available")) is not True:
            reason = "size not available"
        elif constraints.color is not None and color_matches(constraints.color, attrs.get("color")) is not True:
            reason = "different colour"
        elif constraints.min_rating is not None and p.get("rating") is None:
            reason = "rating not available"      # unknown, not "low": the minimum cannot be verified
        elif constraints.min_rating is not None and p["rating"] < constraints.min_rating:
            reason = "rating too low"
        elif constraints.in_stock_only and (p.get("stock_qty") or 0) <= 0:
            reason = "out of stock"

        if reason:
            dropped.append({"product_id": p["product_id"], "reason": reason})
        else:
            kept.append(p)
    return kept, dropped


def sort_products(products: List[Dict[str, Any]], sort_by: SortBy = "relevance") -> List[Dict[str, Any]]:
    """Sort by a catalogue field. Missing prices/ratings always go last."""
    if sort_by == "price_asc":
        return sorted(products, key=lambda p: (p.get("price") is None, p.get("price") or 0))
    if sort_by == "price_desc":
        return sorted(products, key=lambda p: (p.get("price") is None, -(p.get("price") or 0)))
    if sort_by == "rating":
        return sorted(products, key=lambda p: (p.get("rating") is None, -(p.get("rating") or 0)))
    return list(products)


def check_stock(product_id: str, size: Optional[str] = None, color: Optional[str] = None) -> StockResult:
    """Answer 'is it available (in size X / colour Y)?' from the database."""
    product = repo.get_product(product_id)
    if not product:
        return StockResult(product_id=product_id, found=False, message=f"{product_id} is not in our catalogue.")

    attrs = product.get("attributes") or {}
    qty = product.get("stock_qty") or 0
    size_ok = size_matches(size, attrs.get("sizes_available"))
    color_ok = color_matches(color, attrs.get("color"))
    available = qty > 0 and size_ok is not False and color_ok is not False

    parts = [f"{product['name']} ({product_id})"]
    if qty <= 0:
        parts.append(f"is currently out of stock{stock_note(product)}")
    else:
        parts.append(f"has {qty} unit{'s' if qty != 1 else ''} in stock{stock_note(product)}")
    if size is not None:
        if size_ok is None:
            parts.append(f"but size information is not listed")
        else:
            parts.append(f"size {size} is {'available' if size_ok else 'NOT available'}")
            if not size_ok and attrs.get("sizes_available"):
                parts.append("available sizes: " + ", ".join(str(s) for s in attrs["sizes_available"]))
    if color is not None:
        if color_ok is None:
            parts.append("colour is not listed")
        else:
            parts.append(f"colour is {attrs.get('color')}")

    return StockResult(
        product_id=product_id, found=True, in_stock=available, stock_qty=qty,
        size_requested=size, size_available=size_ok, color_requested=color, color_match=color_ok,
        message="; ".join(parts) + ".",
    )


_BASE_FIELDS = ["price", "retail_price", "brand", "category", "subcategory", "rating", "stock_qty", "weight_g"]
# Attribute keys that are identifiers or boilerplate, not something a shopper compares
_SKIP_ATTRS = {"model_id", "style_code", "design_code", "model_number", "sales_package",
               "additional_info", "number_of_contents_in_sales_package"}
MAX_ATTR_FIELDS = 8


def compare_products(product_ids: List[str]) -> ComparisonTable:
    """Side-by-side table of real catalogue fields. Missing values show as 'Not available in catalogue'
    (never as 0 or 'no'). Base fields that no compared product has are left out; attributes present
    in every product come first."""
    products = repo.get_products(product_ids)
    counts: Dict[str, int] = {}
    order: List[str] = []
    for p in products:
        for key, value in (p.get("attributes") or {}).items():
            if key in _SKIP_ATTRS or value in (None, "", []):
                continue
            if key not in counts:
                order.append(key)
            counts[key] = counts.get(key, 0) + 1
    attr_keys = sorted(order, key=lambda k: (-counts[k], order.index(k)))[:MAX_ATTR_FIELDS]
    base = [f for f in _BASE_FIELDS if any(p.get(f) not in (None, "") for p in products) or f == "rating"]
    fields = base + attr_keys

    rows: List[ComparisonRow] = []
    for p in products:
        attrs = p.get("attributes") or {}
        values: Dict[str, Any] = {}
        for field in fields:
            value = p.get(field) if field in _BASE_FIELDS else attrs.get(field)
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            if field == "stock_qty" and value is not None:
                value = f"{value} in stock{stock_note(p)}" if value > 0 else f"Out of stock{stock_note(p)}"
            values[field] = NOT_AVAILABLE if value is None or value == "" else value
        rows.append(ComparisonRow(product_id=p["product_id"], name=p["name"], values=values))
    return ComparisonTable(fields=fields, rows=rows)
