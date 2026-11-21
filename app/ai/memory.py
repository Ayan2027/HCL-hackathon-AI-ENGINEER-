"""
Conversation memory (Member 1) - R5 "Remember the conversation".

The session state (SessionState in app/schemas.py) is saved in SQLite after
every turn. It holds the active constraints, the products last shown (in
order, so "the first two" works), and the clarification counter.

merge_constraints() decides, in plain code, whether a new message continues
the previous search ("only Nike", "cheaper ones") or starts a new one.
"""
import re
from typing import List, Optional, Tuple

from app.config import settings
from app.db import repository as repo
from app.schemas import AnswerType, Constraints, Intent, SessionState, TurnRecord

_CHEAPER = re.compile(r"\b(cheaper|less expensive|lower price|kam price|kam daam|aur saste|aur sasta|"
                      r"thoda sasta|sasta wala|saste wale)\b|और सस्ते|और सस्ता|सस्ते वाले", re.I)
_COSTLIER = re.compile(r"\b(costlier|more expensive|pricier|higher price|aur mehnge|mehnge wale)\b|और महंगे", re.I)
_ANY_BRAND = re.compile(r"\b(any brand|all brands|koi bhi brand|kisi bhi brand)\b|कोई भी ब्रांड", re.I)
_ANY_PRICE = re.compile(r"\b(any price|no budget|budget no issue|price no issue|koi bhi price)\b", re.I)


class SessionOwnershipError(Exception):
    """The session_id was created by a different customer (R9 data isolation)."""


def load_state(session_id: str, customer_id: str) -> SessionState:
    row = repo.get_session(session_id)
    if row is None:
        return SessionState(customer_id=customer_id)
    if row["customer_id"] != customer_id:
        raise SessionOwnershipError(session_id)
    try:
        state = SessionState.model_validate(row["state"])
    except Exception:  # noqa: BLE001 - corrupt state: start clean rather than crash
        state = SessionState(customer_id=customer_id)
    state.customer_id = customer_id
    _drop_stale_catalogue_refs(state)
    return state


def _drop_stale_catalogue_refs(state: SessionState) -> None:
    """After a catalogue change, a saved session may point at products or categories that are no
    longer sold. Drop those so they can never resurface in results or filters."""
    if state.last_result_ids:
        active = {p["product_id"] for p in repo.get_products(state.last_result_ids)}
        state.last_result_ids = [pid for pid in state.last_result_ids if pid in active]
    if state.constraints.category and state.constraints.category not in repo.list_categories():
        state.constraints.category = None


def save_state(session_id: str, state: SessionState) -> None:
    repo.save_session(session_id, state.customer_id, state)


def _overlay(base: Constraints, new: Constraints, keep_keywords: bool) -> Constraints:
    merged = base.model_copy(deep=True)
    if new.category:
        merged.category = new.category
    if new.subcategory:
        merged.subcategory = new.subcategory
    if new.budget_min is not None:
        merged.budget_min = new.budget_min
    if new.budget_max is not None:
        merged.budget_max = new.budget_max
    if new.brands:
        merged.brands = list(new.brands)
        merged.exclude_brands = [b for b in merged.exclude_brands if b not in new.brands]
    if new.exclude_brands:
        merged.exclude_brands = list(dict.fromkeys(merged.exclude_brands + new.exclude_brands))
        merged.brands = [b for b in merged.brands if b not in new.exclude_brands]
    if new.size:
        merged.size = new.size
    if new.color:
        merged.color = new.color
    if new.occasion:
        merged.occasion = new.occasion
    if new.min_rating is not None:
        merged.min_rating = new.min_rating
    if new.sort_by != "relevance":
        merged.sort_by = new.sort_by
    if keep_keywords:
        merged.keywords = list(dict.fromkeys(base.keywords + new.keywords))[:10]
    else:
        merged.keywords = list(new.keywords)
    return merged


def merge_constraints(
    state: SessionState, intent: Intent, message: str, continuing_clarification: bool
) -> Tuple[Constraints, List[str]]:
    """Combine remembered constraints with the new message. Returns (constraints, notes for the trace)."""
    prev = state.constraints
    new = intent.constraints
    notes: List[str] = []

    follow_up = (
        intent.intent in {"refine", "compare", "product_fact", "stock_check"}
        or continuing_clarification
        or (intent.refers_to_previous and not new.category)
        or (intent.intent == "search" and not new.category and not new.keywords and not new.occasion)
    )
    switched_category = bool(new.category and prev.category and new.category != prev.category)

    if follow_up and not switched_category:
        merged = _overlay(prev, new, keep_keywords=True)
        notes.append("continued previous search")
    else:
        merged = new.model_copy(deep=True)
        notes.append("new search")

    # Relative price ("show cheaper ones") is computed by code from the prices last shown
    last_prices = [p["price"] for p in repo.get_products(state.last_result_ids) if p.get("price") is not None]
    if last_prices and _CHEAPER.search(message):
        merged.budget_max = min(last_prices) - 1
        merged.budget_min = None
        merged.sort_by = "price_asc"
        notes.append(f"cheaper than ₹{min(last_prices):,.0f} (cheapest shown)")
    elif last_prices and _COSTLIER.search(message):
        merged.budget_min = max(last_prices) + 1
        merged.budget_max = None
        notes.append(f"pricier than ₹{max(last_prices):,.0f}")
    if _ANY_BRAND.search(message):
        merged.brands, merged.exclude_brands = [], []
        notes.append("brand filter cleared")
    if _ANY_PRICE.search(message):
        merged.budget_min = merged.budget_max = None
        notes.append("budget cleared")
    return merged, notes


def resolve_referenced_products(intent: Intent, state: SessionState) -> List[str]:
    """Turn 'the first two', 'P-1004', 'the blue one', 'it' into product IDs (in order)."""
    ids: List[str] = list(intent.product_ids_mentioned)
    last = state.last_result_ids
    for pos in intent.referenced_positions:
        index = len(last) - 1 if pos == -1 else pos - 1
        if 0 <= index < len(last) and last[index] not in ids:
            ids.append(last[index])
    if not ids and last and intent.constraints.color:
        for product in repo.get_products(last):
            colour = str((product.get("attributes") or {}).get("color", "")).lower()
            if intent.constraints.color.lower() in colour:
                ids.append(product["product_id"])
                break
    if not ids and last and intent.refers_to_previous and intent.intent in {"product_fact", "stock_check"}:
        ids.append(last[0])
    return ids


def record_turn(
    state: SessionState,
    user_message: str,
    answer: str,
    answer_type: AnswerType,
    product_ids: Optional[List[str]] = None,
) -> None:
    state.history.append(TurnRecord(role="user", text=user_message[:500]))
    state.history.append(TurnRecord(role="assistant", text=answer[:500], answer_type=answer_type,
                                    product_ids=product_ids or []))
    state.history = state.history[-2 * settings.history_turns:]
    state.turn_count += 1


def last_answer_type(state: SessionState) -> str:
    for turn in reversed(state.history):
        if turn.role == "assistant":
            return turn.answer_type or ""
    return ""
