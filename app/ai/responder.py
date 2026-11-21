"""
Grounded answer writing (Member 1) - R4 "Give a reason", R6 "Never make things up".

1. Code builds a FACT SHEET from database rows (prices formatted by code).
2. Code writes a template reason for every product (always correct).
3. The LLM may rewrite the intro and reasons in the shopper's language,
   using only the fact sheet.
4. grounding_check() rejects any LLM text that mentions a product ID, price
   or discount that is not in the fact sheet -> the template is used instead.
"""
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.ai import llm_client
from app.ai.language import phrase, suggestion
from app.ai.prompts import ANSWER_SYSTEM, build_answer_user_prompt
from app.ai.safety import looks_like_injection, sanitize_catalog_text
from app.config import PRODUCT_ID_IN_TEXT
from app.schemas import ComparisonTable, Constraints, ProductCard, SourceRef, StockResult
from app.tools.catalog_tools import SIMULATED_NOTE, stock_note

_PRICE_IN_TEXT = re.compile(r"(?:₹|rs\.?\s?|inr\s?)\s*([\d,]+(?:\.\d+)?)", re.I)
_PID_IN_TEXT = re.compile(PRODUCT_ID_IN_TEXT, re.I)
# Claims the LLM may make only if the same word appears in the facts it was given
_UNVERIFIED_TERMS = re.compile(r"\b(warranty|guarantee|guaranteed|delivery|delivered|shipping|emi|"
                               r"return policy|replacement|genuine|original product|bestseller|best-seller)\b", re.I)
# Attributes shown on cards: descriptive facts, not identifiers or boilerplate
_CARD_ATTR_SKIP = {"model_id", "style_code", "design_code", "model_number", "sales_package", "additional_info",
                   "number_of_contents_in_sales_package", "warranty_summary", "covered_in_warranty",
                   "not_covered_in_warranty", "warranty_service_type"}
MAX_CARD_ATTRS = 5
_INVENTED_DEALS = re.compile(r"\b(discount|% off|percent off|offer|coupon|cashback|sale price|free delivery)\b", re.I)

WHY_TEXT = {
    "over budget": {"en": "{n} over your budget", "hinglish": "{n} budget se upar", "hi": "{n} बजट से ऊपर"},
    "out of stock": {"en": "{n} out of stock", "hinglish": "{n} out of stock", "hi": "{n} स्टॉक में नहीं"},
    "different brand": {"en": "{n} from other brands", "hinglish": "{n} doosre brand ke", "hi": "{n} दूसरे ब्रांड के"},
    "size not available": {"en": "{n} not in your size", "hinglish": "{n} aapke size mein nahi", "hi": "{n} आपके साइज़ में नहीं"},
    "different colour": {"en": "{n} in other colours", "hinglish": "{n} doosre colour mein", "hi": "{n} दूसरे रंग में"},
    "price not listed": {"en": "{n} without a listed price", "hinglish": "{n} ka price listed nahi", "hi": "{n} की कीमत उपलब्ध नहीं"},
    "below minimum price": {"en": "{n} below your minimum price", "hinglish": "{n} minimum price se kam", "hi": "{n} न्यूनतम कीमत से कम"},
    "rating too low": {"en": "{n} rated lower than you asked", "hinglish": "{n} ki rating kam hai", "hi": "{n} की रेटिंग कम"},
    "rating not available": {"en": "{n} without a rating", "hinglish": "{n} ki rating available nahi", "hi": "{n} की रेटिंग उपलब्ध नहीं"},
    "different category": {"en": "{n} in other categories", "hinglish": "{n} doosri category mein", "hi": "{n} दूसरी श्रेणी में"},
}


def format_inr(amount: Optional[float]) -> str:
    """1234567 -> '₹12,34,567' (Indian digit grouping)."""
    if amount is None:
        return "price not listed"
    rupees = int(round(amount))
    s = str(rupees)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups + [tail])
    return f"₹{s}"


# ---------------------------------------------------------------------------
# Facts and template reasons (pure code)
# ---------------------------------------------------------------------------
def template_reason(product: Dict[str, Any], constraints: Constraints) -> str:
    parts: List[str] = []
    price = product.get("price")
    if price is not None and constraints.budget_max is not None:
        parts.append(f"{format_inr(price)}, within your {format_inr(constraints.budget_max)} budget")
    elif price is not None:
        parts.append(f"priced at {format_inr(price)}")
    rank = product.get("_rank", {})
    if product.get("best_for") and rank.get("matched_terms"):
        parts.append(f"made for {product['best_for']}")
    elif product.get("best_for"):
        parts.append(f"good for {product['best_for']}")
    elif rank.get("matched_terms"):
        parts.append("matches " + ", ".join(f"'{t}'" for t in rank["matched_terms"][:2]))
    elif product.get("subcategory"):
        parts.append(str(product["subcategory"]).split(">")[-1].strip())
    if product.get("rating") is not None:
        parts.append(f"rated {product['rating']}/5")
    if constraints.size:
        parts.append(f"size {constraints.size} available")
    qty = product.get("stock_qty") or 0
    if 0 < qty <= 3:
        parts.append(f"only {qty} left in stock{stock_note(product)}")
    if rank.get("personal") and product.get("brand"):
        parts.append(f"a brand you have bought or liked before ({product['brand']})")
    text = "; ".join(parts[:4]) or "matches your search"
    return text[0].upper() + text[1:]


def product_fact_sheet(product: Dict[str, Any], constraints: Constraints) -> Dict[str, Any]:
    """The ONLY product information the answer-writing LLM is allowed to use."""
    # Attribute values are catalogue text too: strip instruction-like sentences
    attrs = {k: (sanitize_catalog_text(v) if isinstance(v, str) else v)
             for k, v in (product.get("attributes") or {}).items()}
    qty = product.get("stock_qty") or 0
    return {
        "product_id": product["product_id"],
        "name": product["name"],
        "brand": product.get("brand") or "not listed",
        "category": product.get("category") or "not listed",
        "subcategory": product.get("subcategory") or "not listed",
        "price": format_inr(product.get("price")),
        "mrp": format_inr(product.get("retail_price")) if product.get("retail_price") is not None else "not listed",
        "rating": product.get("rating") if product.get("rating") is not None else "not listed",
        "stock": (f"{qty} in stock" if qty > 0 else "out of stock") + stock_note(product),
        "attributes": attrs,
        "best_for": product.get("best_for") or "not listed",
        "description": sanitize_catalog_text(product.get("description")) or "",
        "review_summary": sanitize_catalog_text(product.get("review_summary")) or "",
        "why_it_fits_facts": template_reason(product, constraints),
    }


def card_attributes(product: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in (product.get("attributes") or {}).items():
        if key in _CARD_ATTR_SKIP or value in (None, "", []) or len(out) >= MAX_CARD_ATTRS:
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        text = sanitize_catalog_text(str(value)) or ""
        if text and len(text) <= 60:
            out[key] = text
    return out


def _valid_image(url: Any) -> Optional[str]:
    url = str(url or "").strip()
    return url if url.lower().startswith(("http://", "https://")) else None


def to_card(product: Dict[str, Any], reason: str) -> ProductCard:
    return ProductCard(
        product_id=product["product_id"], name=product["name"], brand=product.get("brand"),
        category=product.get("category"), price=product.get("price"),
        in_stock=(product.get("stock_qty") or 0) > 0, stock_qty=product.get("stock_qty") or 0,
        rating=product.get("rating"), reason=reason,
        subcategory=product.get("subcategory"), retail_price=product.get("retail_price"),
        image_url=_valid_image(product.get("image_url")),
        stock_is_simulated=bool(product.get("stock_is_simulated")),
        attributes=card_attributes(product),
    )


def to_sources(products: List[Dict[str, Any]], fields: List[str]) -> List[SourceRef]:
    return [SourceRef(product_id=p["product_id"], fields=fields) for p in products]


# ---------------------------------------------------------------------------
# Grounding guard
# ---------------------------------------------------------------------------
def _allowed_numbers(products: List[Dict[str, Any]], constraints: Constraints, extra: Set[float]) -> Set[float]:
    allowed = set(extra)
    for p in products:
        for value in (p.get("price"), p.get("stock_qty"), p.get("rating"), p.get("rating_count"),
                      p.get("retail_price")):
            if isinstance(value, (int, float)):
                allowed.add(float(value))
    for value in (constraints.budget_min, constraints.budget_max):
        if value is not None:
            allowed.add(float(value))
    return allowed


def grounding_check(text: str, allowed_ids: Set[str], allowed_numbers: Set[float],
                    facts_text: Optional[str] = None) -> Tuple[bool, str]:
    """True if every product ID and rupee amount in text exists in the facts, and claims such as
    warranty / delivery appear only when the facts mention them (facts_text)."""
    if not text:
        return True, ""
    allowed_ids = {i.upper() for i in allowed_ids}
    for pid in _PID_IN_TEXT.findall(text):
        if pid.upper() not in allowed_ids:
            return False, f"unknown product id {pid}"
    for raw in _PRICE_IN_TEXT.findall(text):
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if value not in allowed_numbers:
            return False, f"price ₹{raw} not in facts"
    if _INVENTED_DEALS.search(text):
        return False, "mentions a discount/offer not in the catalogue"
    if facts_text is not None:
        for m in _UNVERIFIED_TERMS.finditer(text):
            if m.group(1).lower() not in facts_text:
                return False, f"mentions '{m.group(1)}' which is not in the facts"
    if looks_like_injection(text):
        return False, "instruction-like text in output"
    return True, ""


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
def _template_intro(language: str, n: int, constraints: Constraints) -> str:
    extra = ""
    if constraints.budget_max is not None:
        extra = {"en": f" under {format_inr(constraints.budget_max)}",
                 "hinglish": f" ({format_inr(constraints.budget_max)} se kam)",
                 "hi": f" ({format_inr(constraints.budget_max)} से कम)"}.get(language, "")
    verb = "is" if n == 1 else "are"
    return phrase("found", language, n=n, s="" if n == 1 else "s", verb=verb, extra=extra)


def write_recommendation(
    message: str, language: str, products: List[Dict[str, Any]], constraints: Constraints
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """Returns (answer, reasons_by_id, info_for_trace)."""
    reasons = {p["product_id"]: template_reason(p, constraints) for p in products}
    answer = _template_intro(language, len(products), constraints)
    info: Dict[str, Any] = {"writer": "template"}

    facts = {
        "shopper_constraints": constraints.model_dump(exclude_defaults=True),
        "products_in_display_order": [product_fact_sheet(p, constraints) for p in products],
    }
    try:
        data = llm_client.chat_json(
            ANSWER_SYSTEM,
            build_answer_user_prompt("Recommend these products (already chosen and ranked by code).",
                                     language, message, facts),
        )
    except llm_client.LLMUnavailable as exc:
        info["llm"] = f"skipped: {exc}"
        return answer, reasons, info

    allowed_ids = {p["product_id"] for p in products}
    allowed_nums = _allowed_numbers(products, constraints, set())
    facts_text = json.dumps(facts, ensure_ascii=False).lower()
    llm_answer = str(data.get("answer") or "").strip()
    ok, why = grounding_check(llm_answer, allowed_ids, allowed_nums, facts_text)
    if llm_answer and ok:
        answer = llm_answer
        info["writer"] = "llm"
    elif llm_answer:
        info["guard_rejected_answer"] = why

    llm_reasons = data.get("reasons") if isinstance(data.get("reasons"), dict) else {}
    rejected = {}
    for pid in allowed_ids:
        text = str(llm_reasons.get(pid) or "").strip()
        if not text:
            continue
        ok, why = grounding_check(text, allowed_ids, _allowed_numbers(
            [p for p in products if p["product_id"] == pid], constraints, set()), facts_text)
        if ok:
            reasons[pid] = text
        else:
            rejected[pid] = why
    if rejected:
        info["guard_rejected_reasons"] = rejected
    return answer, reasons, info


def comparison_verdict(table: ComparisonTable, products: List[Dict[str, Any]]) -> str:
    """Plain verdict computed by code from catalogue fields."""
    priced = [p for p in products if p.get("price") is not None]
    rated = [p for p in products if p.get("rating") is not None]
    parts = []
    if priced:
        cheapest = min(priced, key=lambda p: p["price"])
        parts.append(f"{cheapest['name']} ({cheapest['product_id']}) is the cheapest at {format_inr(cheapest['price'])}")
    if rated:
        best = max(rated, key=lambda p: p["rating"])
        parts.append(f"{best['name']} ({best['product_id']}) has the highest rating ({best['rating']}/5)")
    unrated = [p for p in products if p.get("rating") is None]
    if unrated and rated:
        parts.append("rating not available in catalogue for " + ", ".join(p["product_id"] for p in unrated))
    elif unrated:
        parts.append("no rating is available in the catalogue for these products")
    out = [p for p in products if (p.get("stock_qty") or 0) <= 0]
    if out:
        parts.append(", ".join(f"{p['name']} ({p['product_id']})" for p in out)
                     + f" is out of stock ({SIMULATED_NOTE})")
    return ("; ".join(parts) + ".") if parts else "Both options are similar on the listed fields."


def write_comparison(
    message: str, language: str, table: ComparisonTable, products: List[Dict[str, Any]]
) -> Tuple[str, Dict[str, Any]]:
    verdict = comparison_verdict(table, products)
    answer = f"{phrase('compare_intro', language)} {verdict}"
    info: Dict[str, Any] = {"writer": "template", "verdict_facts": verdict}
    facts = {
        "comparison_table": table.model_dump(),
        "verdict_from_code": verdict,
        "products": [product_fact_sheet(p, Constraints()) for p in products],
    }
    try:
        data = llm_client.chat_json(
            ANSWER_SYSTEM,
            build_answer_user_prompt(
                "Write a one-sentence 'answer' introducing the comparison and a 'verdict' that restates "
                "verdict_from_code in plain words (you may add which product suits what need, from FACTS).",
                language, message, facts),
        )
    except llm_client.LLMUnavailable as exc:
        info["llm"] = f"skipped: {exc}"
        return answer, info
    text = " ".join(str(data.get(k) or "").strip() for k in ("answer", "verdict")).strip()
    allowed_ids = {p["product_id"] for p in products}
    ok, why = grounding_check(text, allowed_ids, _allowed_numbers(products, Constraints(), set()),
                              json.dumps(facts, ensure_ascii=False).lower())
    if text and ok:
        info["writer"] = "llm"
        return text, info
    info["guard_rejected_answer"] = why or "empty"
    return answer, info


def not_found_text(language: str, dropped: List[Dict[str, Any]], constraints: Constraints) -> str:
    counts: Dict[str, int] = {}
    for d in dropped:
        if d["reason"] in WHY_TEXT:
            counts[d["reason"]] = counts.get(d["reason"], 0) + 1
    why = ""
    if counts:
        pieces = [WHY_TEXT[r].get(language, WHY_TEXT[r]["en"]).format(n=n) for r, n in counts.items()]
        label = {"en": "close matches: ", "hinglish": "milte-julte products: ", "hi": "मिलते-जुलते उत्पाद: "}
        why = f" ({label.get(language, label['en'])}{', '.join(pieces)})"
    return phrase("not_found", language, why=why)


def widen_suggestions(language: str, dropped_products: List[Dict[str, Any]], constraints: Constraints) -> List[str]:
    """Code-made chips that would actually return results (R6: offer to widen the search)."""
    chips: List[str] = []
    over = [p for p in dropped_products if p.get("_drop_reason") == "over budget" and p.get("price")
            and (p.get("stock_qty") or 0) > 0]
    if over and constraints.budget_max is not None:
        cheapest_over = min(p["price"] for p in over)
        chips.append(suggestion("widen_budget", language, amount=f"{int(cheapest_over):,}"))
    if constraints.brands:
        chips.append(suggestion("any_brand", language))
    return chips[:3]


def recommendation_suggestions(language: str, products: List[Dict[str, Any]], constraints: Constraints,
                               pool: List[Dict[str, Any]]) -> List[str]:
    """Follow-up chips built from filters the shopper has NOT used yet."""
    chips: List[str] = []
    if len(products) >= 2:
        chips.append(suggestion("compare", language))
    priced = [p["price"] for p in products if p.get("price")]
    if priced and len(pool) > len(products):
        chips.append(suggestion("cheaper", language))
    if not constraints.brands:
        brands = [p.get("brand") for p in pool if p.get("brand")]
        if len(set(brands)) >= 2:
            chips.append(suggestion("only_brand", language, brand=brands[0]))
    return chips[:3]


def stock_text(result: StockResult, product: Dict[str, Any], language: str) -> str:
    """Stock answer written by code in the shopper's language (no LLM, no guessing)."""
    if language == "en" or not result.found:
        return result.message
    sizes = ", ".join(str(s) for s in (product.get("attributes") or {}).get("sizes_available", []))
    name = f"{product['name']} ({product['product_id']})"
    if language == "hinglish":
        parts = [f"{name} abhi stock mein hai ({result.stock_qty} piece)" if result.stock_qty > 0
                 else f"{name} abhi out of stock hai"]
        if product.get("stock_is_simulated"):
            parts[0] += " (simulated demo inventory)"
        if result.size_requested is not None:
            if result.size_available is None:
                parts.append("size ki jaankari listed nahi hai")
            elif result.size_available:
                parts.append(f"size {result.size_requested} available hai")
            else:
                parts.append(f"size {result.size_requested} available nahi hai (available sizes: {sizes})")
        return ", ".join(parts) + "."
    parts = [f"{name} अभी स्टॉक में है ({result.stock_qty} पीस)" if result.stock_qty > 0
             else f"{name} अभी स्टॉक में नहीं है"]
    if product.get("stock_is_simulated"):
        parts[0] += " (सिम्युलेटेड डेमो इन्वेंटरी)"
    if result.size_requested is not None:
        if result.size_available is None:
            parts.append("साइज़ की जानकारी उपलब्ध नहीं है")
        elif result.size_available:
            parts.append(f"साइज़ {result.size_requested} उपलब्ध है")
        else:
            parts.append(f"साइज़ {result.size_requested} उपलब्ध नहीं है (उपलब्ध साइज़: {sizes})")
    return ", ".join(parts) + "।"
