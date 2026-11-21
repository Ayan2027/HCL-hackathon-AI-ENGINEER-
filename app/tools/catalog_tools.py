"""
Deterministic catalogue tools (Member 2) - R7: "Let code do the exact work".

Price filters, brand/size/colour checks, stock checks, sorting and comparison
tables are computed here from database values. The LLM only explains results.
"""
from typing import Any, Dict, List, Optional, Tuple

from app.db import repository as repo
from app.schemas import ComparisonRow, ComparisonTable, Constraints, SortBy, StockResult
from app.tools.normalize import color_matches, size_matches


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
        elif constraints.min_rating is not None and (p.get("rating") or 0) < constraints.min_rating:
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
        parts.append("is currently out of stock")
    else:
        parts.append(f"has {qty} unit{'s' if qty != 1 else ''} in stock")
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


_BASE_FIELDS = ["price", "brand", "rating", "stock_qty", "weight_g"]


def compare_products(product_ids: List[str]) -> ComparisonTable:
    """Side-by-side table of real catalogue fields. Missing values show as 'not listed'."""
    products = repo.get_products(product_ids)
    attr_keys: List[str] = []
    for p in products:
        for key in (p.get("attributes") or {}):
            if key not in attr_keys:
                attr_keys.append(key)
    fields = _BASE_FIELDS + attr_keys[:8]

    rows: List[ComparisonRow] = []
    for p in products:
        attrs = p.get("attributes") or {}
        values: Dict[str, Any] = {}
        for field in fields:
            value = p.get(field) if field in _BASE_FIELDS else attrs.get(field)
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            values[field] = "not listed" if value is None or value == "" else value
        rows.append(ComparisonRow(product_id=p["product_id"], name=p["name"], values=values))
    return ComparisonTable(fields=fields, rows=rows)
