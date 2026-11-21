"""
Ranking (Member 1) - "Rank by how well a product fits the request, not by popularity".

final = retrieval_fit                      (RRF score, 0..1)
      + 0.10 * coverage, where coverage = IDF-weighted share of the shopper's keywords/occasion
             words found in the product's best_for, name, subcategory or attribute values
      + rating adjustment                  0.05 * (rating - 3) / 2, i.e. -0.05..+0.05;
                                           a MISSING rating is unknown -> 0 (neither
                                           rewarded nor penalised)
      + 0.05 if THIS shopper bought / prefers the brand (personalisation, own data only)

Hard constraints (budget, brand, stock ...) are applied BEFORE ranking by
tools.apply_filters, so no bonus can bring back a product that breaks them.

Then, if the shopper asked for "cheap" / "best rated", the 10 best-fitting
products are re-sorted by that field with code (tools.sort_products).
"""
import math
import re
from typing import Any, Dict, List, Optional

from app.schemas import Constraints
from app.tools.catalog_tools import sort_products
from app.db.repository import FTS_SKIP_ATTRIBUTES
from app.tools.normalize import attributes_to_text

FIT_BONUS = 0.10
RATING_WEIGHT = 0.05
PERSONAL_BONUS = 0.05
SORT_POOL = 10


def personal_brands(customer: Optional[Dict[str, Any]], orders: List[Dict[str, Any]]) -> List[str]:
    """Brands this shopper bought or said they like. Uses ONLY the current shopper's data (R9)."""
    brands = [o.get("brand") for o in orders if o.get("brand")]
    if customer:
        brands += customer.get("preferred_brands") or []
    return list(dict.fromkeys(b.lower() for b in brands if b))


def _fit_terms(constraints: Constraints) -> List[str]:
    terms = [k.lower() for k in constraints.keywords if len(k) > 2]
    if constraints.occasion:
        terms += [w for w in constraints.occasion.lower().split() if len(w) > 3 and w not in {"gift"}]
    return terms


def rating_adjustment(rating: Optional[float]) -> float:
    """Centred on 3/5 so that 'no rating' (0 adjustment) is treated as unknown, not as bad."""
    if rating is None:
        return 0.0
    return round(RATING_WEIGHT * (float(rating) - 3.0) / 2.0, 4)


def _has_term(term: str, text: str) -> bool:
    """Whole-word match, singular or plural ('cover' matches 'covers' but not 'covered')."""
    stem = re.escape(term[:-1] if term.endswith("s") and len(term) > 3 else term)
    return re.search(rf"(?<![a-z0-9]){stem}(?:s|es)?(?![a-z0-9])", text) is not None


def _fit_text(product: Dict[str, Any]) -> str:
    return " ".join(str(x) for x in [
        product.get("best_for"), product.get("name"), product.get("subcategory"),
        attributes_to_text({k: v for k, v in (product.get("attributes") or {}).items()
                            if k not in FTS_SKIP_ATTRIBUTES}),
    ] if x).lower()


def rank_products(
    products: List[Dict[str, Any]],
    constraints: Constraints,
    customer: Optional[Dict[str, Any]] = None,
    orders: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Adds '_rank' info to each product and returns them best first."""
    liked = personal_brands(customer, orders or [])
    terms = _fit_terms(constraints)
    texts = {p["product_id"]: _fit_text(p) for p in products}
    # Rare words count more than words every candidate shares ("cover" vs "mobile")
    n = len(products)
    weights = {t: math.log((n + 1) / (sum(1 for x in texts.values() if _has_term(t, x)) + 0.5)) for t in terms}
    total = sum(weights.values())
    ranked = []
    for p in products:
        retrieval = p.get("_retrieval", {})
        fit = float(retrieval.get("score", 0.0))
        text = texts[p["product_id"]]
        matched = [t for t in terms if _has_term(t, text)]
        coverage = (sum(weights[t] for t in matched) / total) if total > 0 else (1.0 if matched else 0.0)
        fit_bonus = round(FIT_BONUS * coverage, 4)
        rating_bonus = rating_adjustment(p.get("rating"))
        is_personal = bool(p.get("brand")) and p["brand"].lower() in liked
        personal_bonus = PERSONAL_BONUS if is_personal else 0.0
        final = round(fit + fit_bonus + rating_bonus + personal_bonus, 4)
        ranked.append({**p, "_rank": {
            "final": final, "fit": fit, "fit_bonus": fit_bonus, "matched_terms": matched,
            "coverage": round(coverage, 3),
            "rating_bonus": round(rating_bonus, 4), "personal": is_personal,
        }})
    ranked.sort(key=lambda x: x["_rank"]["final"], reverse=True)

    if constraints.sort_by != "relevance":
        # "cheapest" / "best rated" re-sorts only the products that match the request best,
        # so a cheaper but less relevant item (headphones for "cheap mobile cover") cannot jump ahead
        pool = ranked[:SORT_POOL]
        best = max((p["_rank"]["coverage"] for p in pool), default=0.0)
        top = [p for p in pool if p["_rank"]["coverage"] >= best - 1e-9]
        rest = [p for p in pool if p["_rank"]["coverage"] < best - 1e-9]
        ranked = sort_products(top, constraints.sort_by) + rest + ranked[SORT_POOL:]
    return ranked
