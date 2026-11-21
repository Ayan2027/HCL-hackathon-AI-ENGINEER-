"""
Ranking (Member 1) - "Rank by how well a product fits the request, not by popularity".

final = retrieval_fit                      (hybrid score, 0..1)
      + 0.10 if the product's best_for matches the shopper's keywords or occasion
      + 0.05 * rating / 5                  (tie-breaker only)
      + 0.05 if THIS shopper bought / prefers the brand (personalisation, own data only)

Before scoring, product_type_gate() keeps only the product type the shopper named
("laptop for gaming" can never return a gaming phone).
Then, if the shopper asked for "cheap" / "best rated", the 10 best-fitting
products are re-sorted by that field with code (tools.sort_products).
"""
from typing import Any, Dict, List, Optional, Tuple

from app.ai.intent import CATEGORY_WORDS
from app.schemas import Constraints
from app.tools.catalog_tools import sort_products

# Words that name a product type ("laptop", "speaker", "shoes"), taken from the intent vocabulary
PRODUCT_NOUNS = {w for words in CATEGORY_WORDS.values() for w in words}

FIT_BONUS = 0.10
RATING_WEIGHT = 0.05
PERSONAL_BONUS = 0.05
SORT_POOL = 10


def _stem(word: str) -> str:
    word = word.lower()
    for suffix in ("es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def product_type_gate(products: List[Dict[str, Any]], constraints: Constraints) -> Tuple[List[Dict[str, Any]], List[str]]:
    """If the shopper named a product type ("laptop"), keep only products of that type.
    Products are scored by how many of the named types appear in their name / subcategory / best_for;
    only the best-matching group is kept. Returns (products, nouns used)."""
    nouns = [k for k in constraints.keywords if k.lower() in PRODUCT_NOUNS]
    if not nouns or not products:
        return products, []
    stems = {_stem(n) for n in nouns}

    def hits(p: Dict[str, Any]) -> int:
        text = " ".join(str(p.get(f) or "") for f in ("name", "subcategory", "best_for")).lower()
        return sum(1 for s in stems if s in text)

    best = max(hits(p) for p in products)
    if best == 0:
        return products, []
    return [p for p in products if hits(p) == best], sorted(stems)


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


def rank_products(
    products: List[Dict[str, Any]],
    constraints: Constraints,
    customer: Optional[Dict[str, Any]] = None,
    orders: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Adds '_rank' info to each product and returns them best first."""
    liked = personal_brands(customer, orders or [])
    terms = _fit_terms(constraints)
    ranked = []
    for p in products:
        retrieval = p.get("_retrieval", {})
        fit = float(retrieval.get("score", 0.0))
        text = str(p.get("best_for") or "").lower()
        matched = [t for t in terms if t in text]
        fit_bonus = FIT_BONUS if matched else 0.0
        rating_bonus = RATING_WEIGHT * ((p.get("rating") or 0) / 5)
        is_personal = bool(p.get("brand")) and p["brand"].lower() in liked
        personal_bonus = PERSONAL_BONUS if is_personal else 0.0
        final = round(fit + fit_bonus + rating_bonus + personal_bonus, 4)
        ranked.append({**p, "_rank": {
            "final": final, "fit": fit, "fit_bonus": fit_bonus, "matched_terms": matched,
            "rating_bonus": round(rating_bonus, 4), "personal": is_personal,
        }})
    ranked.sort(key=lambda x: x["_rank"]["final"], reverse=True)

    if constraints.sort_by != "relevance":
        pool = sort_products(ranked[:SORT_POOL], constraints.sort_by)
        ranked = pool + ranked[SORT_POOL:]
    return ranked
