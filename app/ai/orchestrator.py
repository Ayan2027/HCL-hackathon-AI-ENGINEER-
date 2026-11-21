"""
AI orchestrator (Member 1) - the single entry point for POST /chat.

    handle_chat(request) -> ChatResponse

Pipeline (each step is recorded in the trace, R10):
  1  load session memory (session must belong to this customer_id)      code
  2  safety guard: privacy + prompt injection                           code
  3  understand the message -> Intent                                   LLM (rules fallback)
  4  route by intent: compare / stock / fact / my_orders / search ...   code
  5  merge with memory, maybe ask a clarifying question (max 2)         code
  6  hybrid retrieval (FTS5 + FAISS) + relevance gate                   code
  7  exact filters: budget, brand, size, colour, stock                  code (tools)
  8  rank by fit (+ rating tie-break, + own-history boost)              code
  9  write the answer from facts, grounding guard                       LLM (template fallback)
 10  save memory + trace, return JSON
"""
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from app.ai import clarify, intent as intent_mod, memory, responder
from app.ai.language import detect_language, phrase
from app.ai.retrieval import build_query_text, hybrid_retrieve
from app.ai.ranking import product_type_gate, rank_products
from app.ai.safety import check_message
from app.ai.vector_index import get_vector_index
from app.config import settings
from app.db import repository as repo
from app.schemas import (
    AnswerType,
    ChatRequest,
    ChatResponse,
    Constraints,
    Intent,
    SessionState,
    SourceRef,
    ToolCall,
    TraceRecord,
)
from app.tools import catalog_tools as tools

logger = logging.getLogger("shopsense.orchestrator")


class Turn:
    """Collects everything that happens in one reply, for the response and the trace."""

    def __init__(self, request: ChatRequest) -> None:
        self.request = request
        self.trace_id = uuid.uuid4().hex[:8]
        self.started = time.perf_counter()
        self.tools: List[ToolCall] = []
        self.retrieved: List[Dict[str, Any]] = []
        self.filters: Dict[str, Any] = {}
        self.intent: Dict[str, Any] = {}
        self.llm_used = False

    def tool(self, name: str, inputs: Dict[str, Any], summary: str, started: float) -> None:
        self.tools.append(ToolCall(name=name, input=inputs, output_summary=summary[:300],
                                   duration_ms=round((time.perf_counter() - started) * 1000, 1)))


# ---------------------------------------------------------------------------
# Public API used by FastAPI (Member 2)
# ---------------------------------------------------------------------------
def startup() -> int:
    """Build or load the FAISS index. Called once when the API starts."""
    products = repo.get_all_products()
    count = get_vector_index().load_or_build(products)
    logger.info("Vector index ready: %d vectors (enabled=%s)", count, get_vector_index().enabled)
    return count


def on_products_changed(product_ids: List[str]) -> bool:
    """Called by POST /ingest so new/updated products are searchable at once (R11).
    Keyword search is already updated by the repository; this updates the vectors."""
    products = repo.get_products(product_ids)
    index = get_vector_index()
    if not index.enabled:
        return False
    if index.size == 0:
        return index.build(repo.get_all_products()) > 0
    return index.upsert(products)


def vector_index_size() -> int:
    return get_vector_index().size


def handle_chat(request: ChatRequest) -> ChatResponse:
    turn = Turn(request)
    language = detect_language(request.message)
    try:
        response, state = _run_pipeline(turn, language)
    except Exception as exc:  # noqa: BLE001 - a bug must never take the API down
        logger.exception("Pipeline error")
        turn.tools.append(ToolCall(name="error", output_summary=f"{type(exc).__name__}: {exc}"))
        response = ChatResponse(trace_id=turn.trace_id, answer_type="not_found",
                                answer=phrase("error", language), language=language)
        state = None
    if state is not None:
        memory.record_turn(state, request.message, response.answer, response.answer_type,
                           [p.product_id for p in response.products])
        memory.save_state(request.session_id, state)
    _save_trace(turn, response)
    return response


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _respond(turn: Turn, answer_type: AnswerType, answer: str, language: str, **extra) -> ChatResponse:
    return ChatResponse(trace_id=turn.trace_id, answer_type=answer_type, answer=answer, language=language, **extra)


def _run_pipeline(turn: Turn, language: str):
    req = turn.request

    # 1. Memory (session bound to one shopper)
    try:
        state = memory.load_state(req.session_id, req.customer_id)
    except memory.SessionOwnershipError:
        turn.tools.append(ToolCall(name="session_guard", output_summary="session belongs to another customer"))
        return _respond(turn, "refusal", phrase("session_owner", language), language), None

    # 2. Safety (code, before any LLM call)
    t0 = time.perf_counter()
    verdict = check_message(req.message, req.customer_id, repo.list_customer_names())
    turn.tool("safety_check", {"message_chars": len(req.message)}, f"{verdict.kind} {verdict.reason}", t0)
    if verdict.blocked:
        key = "privacy" if verdict.kind == "privacy" else "injection"
        state.clarify_count = 0
        return _respond(turn, "refusal", phrase(key, language), language), state

    # 3. Understand
    previous = repo.get_products(state.last_result_ids)
    previous_brief = [{"position": i + 1, "product_id": p["product_id"], "name": p["name"],
                       "brand": p.get("brand"), "price": p.get("price"),
                       "color": (p.get("attributes") or {}).get("color")} for i, p in enumerate(previous)]
    last_type = memory.last_answer_type(state)
    t0 = time.perf_counter()
    intent, source, note = intent_mod.extract_intent(
        req.message, state, language, repo.list_brands(), repo.list_subcategories(), previous_brief, last_type)
    turn.llm_used = source == "llm"
    if language == "en" and intent.language in {"hi", "hinglish"} and source == "llm":
        language = intent.language
    intent.language = language
    turn.intent = {**intent.model_dump(), "source": source, "note": note}
    turn.tool("understand_intent", {"source": source}, f"{intent.intent} {note}".strip(), t0)
    state.language = language

    # 4. Route
    if intent.intent == "chitchat":
        state.clarify_count = 0
        return _respond(turn, "clarification", phrase("greeting", language), language,
                        follow_up_question=phrase("greeting", language)), state
    if intent.intent == "out_of_scope":
        state.clarify_count = 0
        return _respond(turn, "not_found", phrase("out_of_scope", language), language), state
    if intent.intent == "my_orders":
        state.clarify_count = 0
        return _my_orders(turn, language), state
    if intent.intent == "compare":
        state.clarify_count = 0
        return _compare(turn, intent, state, language), state
    if intent.intent in {"stock_check", "product_fact"}:
        state.clarify_count = 0
        return _product_question(turn, intent, state, language), state
    return _search(turn, intent, state, language, continuing=(last_type == "clarification")), state


def _search(turn: Turn, intent: Intent, state: SessionState, language: str, continuing: bool) -> ChatResponse:
    req = turn.request
    merged, notes = memory.merge_constraints(state, intent, req.message, continuing)
    turn.filters = {"constraints": merged.model_dump(), "memory": notes}

    # 5. Clarify if too vague (max 2 in a row, never re-ask)
    plan = clarify.plan_clarification(intent, merged, state, language)
    if plan:
        question, slots = plan
        state.constraints = merged
        state.clarify_count += 1
        state.asked_slots = list(dict.fromkeys(state.asked_slots + slots))
        turn.tools.append(ToolCall(name="clarify", input={"slots": slots},
                                   output_summary=f"question {state.clarify_count}/{settings.max_clarifications}"))
        return _respond(turn, "clarification", question, language, follow_up_question=question)

    state.clarify_count = 0
    state.asked_slots = []
    state.constraints = merged

    # 6. Retrieve
    query = build_query_text(intent.search_query_en if not continuing else "", merged) or req.message
    t0 = time.perf_counter()
    candidates, debug = hybrid_retrieve(query, merged)
    turn.retrieved = [{"product_id": c["product_id"], **c["_retrieval"]} for c in candidates[:15]]
    turn.tool("hybrid_retrieve", {"query": query}, f"{len(candidates)} candidates ({debug['mode']})", t0)

    # 7. Exact filters (code)
    t0 = time.perf_counter()
    kept, dropped = tools.apply_filters(candidates, merged)
    turn.filters["dropped"] = dropped[:20]
    turn.tool("apply_filters", merged.model_dump(exclude_defaults=True),
              f"kept {len(kept)}, dropped {len(dropped)}", t0)

    if not kept:
        # keep state.last_result_ids: the shopper can still say "compare the first two"
        drop_reason = {d["product_id"]: d["reason"] for d in dropped}
        dropped_products = [{**c, "_drop_reason": drop_reason.get(c["product_id"])} for c in candidates]
        chips = responder.widen_suggestions(language, dropped_products, merged)
        return _respond(turn, "not_found", responder.not_found_text(language, dropped, merged), language,
                        suggested_questions=chips)

    # 8. Rank (own history only)
    t0 = time.perf_counter()
    customer = repo.get_customer(req.customer_id)
    orders = repo.get_orders(req.customer_id)
    typed, nouns = product_type_gate(kept, merged)
    if nouns:
        turn.tool("product_type_gate", {"types": nouns}, f"kept {len(typed)} of {len(kept)}", t0)
    ranked = rank_products(typed, merged, customer, orders)
    top = ranked[: settings.top_k]
    turn.tool("rank_products", {"sort_by": merged.sort_by, "personalised_for": req.customer_id},
              ", ".join(f"{p['product_id']}={p['_rank']['final']}" for p in top), t0)

    # 9. Write the answer (LLM, grounded) and build cards from DB values
    t0 = time.perf_counter()
    answer, reasons, info = responder.write_recommendation(req.message, language, top, merged)
    turn.llm_used = turn.llm_used or info.get("writer") == "llm"
    turn.tool("write_answer", {}, str(info), t0)

    state.last_result_ids = [p["product_id"] for p in top]
    return _respond(
        turn, "recommendation", answer, language,
        products=[responder.to_card(p, reasons[p["product_id"]]) for p in top],
        sources=responder.to_sources(top, ["name", "price", "stock_qty", "rating", "best_for", "description"]),
        suggested_questions=responder.recommendation_suggestions(language, top, merged, ranked),
    )


def _compare(turn: Turn, intent: Intent, state: SessionState, language: str) -> ChatResponse:
    ids = memory.resolve_referenced_products(intent, state)
    if len(ids) < 2 and not intent.referenced_positions and len(state.last_result_ids) >= 2 and not ids:
        ids = state.last_result_ids[:2]                    # "compare them"
    if len(ids) < 2 and len(intent.constraints.brands) >= 2:
        # "compare Nike and Puma running shoes": best match per brand
        for brand in intent.constraints.brands[:3]:
            c = intent.constraints.model_copy(update={"brands": [brand]})
            found, _ = hybrid_retrieve(build_query_text(intent.search_query_en, c), c)
            kept, _ = tools.apply_filters(found, c.model_copy(update={"in_stock_only": False}))
            if kept and kept[0]["product_id"] not in ids:
                ids.append(kept[0]["product_id"])
    if len(ids) < 2:
        msg = phrase("compare_need_two", language)
        return _respond(turn, "clarification", msg, language, follow_up_question=msg)

    t0 = time.perf_counter()
    table = tools.compare_products(ids[:4])
    products = repo.get_products(ids[:4])
    turn.retrieved = [{"product_id": pid, "source": "reference"} for pid in ids[:4]]
    turn.tool("compare_products", {"product_ids": ids[:4]}, f"{len(table.rows)} rows x {len(table.fields)} fields", t0)
    answer, info = responder.write_comparison(turn.request.message, language, table, products)
    turn.llm_used = turn.llm_used or info.get("writer") == "llm"
    turn.tool("write_answer", {}, str(info), t0)
    state.last_result_ids = [p["product_id"] for p in products]
    cards = [responder.to_card(p, responder.template_reason(p, Constraints())) for p in products]
    return _respond(turn, "comparison", answer, language, products=cards, comparison=table,
                    sources=responder.to_sources(products, table.fields))


def _attribute_value(product: Dict[str, Any], attribute: Optional[str]) -> Optional[Any]:
    if not attribute:
        return None
    if attribute == "weight":
        return f"{product['weight_g']:.0f} g" if product.get("weight_g") else product.get("weight_raw")
    if attribute in product and attribute not in {"attributes"}:
        return product.get(attribute)
    attrs = product.get("attributes") or {}
    if attribute in attrs:
        return attrs[attribute]
    for key, value in attrs.items():          # fuzzy: "battery" -> "battery_hours"
        if attribute.split("_")[0] in key:
            return value
    return None


def _product_question(turn: Turn, intent: Intent, state: SessionState, language: str) -> ChatResponse:
    ids = memory.resolve_referenced_products(intent, state)
    if not ids and (intent.search_query_en or intent.constraints.keywords):
        found, _ = hybrid_retrieve(build_query_text(intent.search_query_en, intent.constraints), intent.constraints)
        if intent.constraints.brands:
            found = [p for p in found if (p.get("brand") or "").lower() in {b.lower() for b in intent.constraints.brands}] or found
        found, _ = product_type_gate(found, intent.constraints)
        ids = [found[0]["product_id"]] if found else []
    if not ids:
        msg = phrase("which_product", language)
        return _respond(turn, "clarification", msg, language, follow_up_question=msg)

    product = repo.get_product(ids[0])
    if not product:
        return _respond(turn, "not_found", phrase("not_found", language, why=f" ({ids[0]})"), language)
    turn.retrieved = [{"product_id": product["product_id"], "source": "reference"}]
    c = intent.constraints
    fields = ["stock_qty"]

    t0 = time.perf_counter()
    # A question that names a size (or colour) about a product is answered as a stock/variant check
    is_stock = (intent.intent == "stock_check" or bool(c.size)
                or (c.color and intent.attribute_asked in {None, "stock_qty", "color"}))
    if is_stock:
        result = tools.check_stock(product["product_id"], size=c.size, color=c.color)
        turn.tool("check_stock", {"product_id": product["product_id"], "size": c.size, "color": c.color},
                  result.message, t0)
        answer = responder.stock_text(result, product, language)
        fields += ["attributes.sizes_available", "attributes.color"]
    else:
        if not intent.attribute_asked:
            words = {w.lower().strip("?.,!") for w in turn.request.message.split() if len(w) > 3}
            for key in (product.get("attributes") or {}):
                if any(w in key.lower() for w in words):
                    intent.attribute_asked = key
                    break
        value = _attribute_value(product, intent.attribute_asked)
        label = (intent.attribute_asked or "details").replace("_", " ")
        if intent.attribute_asked == "price":
            value = responder.format_inr(product.get("price")) if product.get("price") is not None else None
        if intent.attribute_asked and value is None:
            answer = f"The {label} of {product['name']} ({product['product_id']}) is not listed in our catalogue."
        elif intent.attribute_asked:
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            answer = f"{product['name']} ({product['product_id']}) - {label}: {value}."
        else:
            answer = (f"{product['name']} ({product['product_id']}) by {product.get('brand') or 'an unlisted brand'}"
                      f" costs {responder.format_inr(product.get('price'))}, is rated "
                      f"{product.get('rating') if product.get('rating') is not None else 'not listed'}"
                      f" and has {product.get('stock_qty') or 0} in stock. {product.get('description') or ''}").strip()
        fields += [f"attributes.{intent.attribute_asked}" if intent.attribute_asked else "description", "price"]
        turn.tool("lookup_product_fact", {"product_id": product["product_id"], "attribute": intent.attribute_asked},
                  answer, t0)

    reason = responder.template_reason(product, Constraints())
    return _respond(turn, "product_fact", answer, language,
                    products=[responder.to_card(product, reason)],
                    sources=[SourceRef(product_id=product["product_id"], fields=fields)])


def _my_orders(turn: Turn, language: str) -> ChatResponse:
    t0 = time.perf_counter()
    orders = repo.get_orders(turn.request.customer_id)       # ONLY this shopper (R9)
    turn.tool("get_own_orders", {"customer_id": turn.request.customer_id}, f"{len(orders)} orders", t0)
    if not orders:
        return _respond(turn, "product_fact", phrase("no_orders", language), language)
    lines = [f"{o['order_date']}: {o['product_name']} ({o['product_id']}) x{o['quantity']}"
             + (f" - {responder.format_inr(o['price_paid'])}" if o.get("price_paid") is not None else "")
             for o in orders[:5]]
    answer = phrase("your_orders", language) + " " + "; ".join(lines) + "."
    products = repo.get_products([o["product_id"] for o in orders[:5]])
    return _respond(turn, "product_fact", answer, language,
                    products=[responder.to_card(p, "From your order history") for p in products],
                    sources=responder.to_sources(products, ["orders"]))


def _save_trace(turn: Turn, response: ChatResponse) -> None:
    try:
        repo.save_trace(TraceRecord(
            trace_id=turn.trace_id, session_id=turn.request.session_id, customer_id=turn.request.customer_id,
            message=turn.request.message, language=response.language, intent=turn.intent,
            retrieved=turn.retrieved, filters=turn.filters, tools=turn.tools,
            final_product_ids=[p.product_id for p in response.products], answer_type=response.answer_type,
            answer=response.answer, follow_up_question=response.follow_up_question,
            response=response.model_dump(), latency_ms=round((time.perf_counter() - turn.started) * 1000, 1),
            llm_mode=settings.llm_mode if turn.llm_used else "mock/template",
        ))
    except Exception:  # noqa: BLE001
        logger.exception("Could not save trace %s", turn.trace_id)
