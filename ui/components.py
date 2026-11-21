"""
Reusable Streamlit pieces (Member 3): answer badges, product cards,
comparison table, follow-up chips and the trace viewer.

Everything shown here comes from the API response - the UI adds no facts.
"""
from typing import Any, Callable, Dict, List

import streamlit as st

import api_client

try:                     # pandas is only used for tables; keep the UI usable without it
    import pandas as pd
except Exception:        # noqa: BLE001 - e.g. blocked numpy DLLs on locked-down Windows
    pd = None


def table(data, **kwargs) -> None:
    """Show tabular data with pandas when available, otherwise Streamlit's plain table."""
    if pd is not None:
        stretch(st.dataframe, pd.DataFrame(data), **kwargs)
    else:                # st.table takes no pandas-only options such as hide_index
        stretch(st.table, data)

ANSWER_BADGES = {
    "recommendation": ("🛍️", "Recommendation", "green"),
    "comparison": ("⚖️", "Comparison", "blue"),
    "clarification": ("❓", "Quick question", "orange"),
    "product_fact": ("📋", "Product info", "violet"),
    "not_found": ("🔍", "Not in catalogue", "gray"),
    "refusal": ("🛡️", "Can't help with that", "red"),
}

def stretch(widget, *args, **kwargs):
    """Full-width widget on any Streamlit version (new: width="stretch", old: use_container_width)."""
    try:
        return widget(*args, width="stretch", **kwargs)
    except TypeError:
        return widget(*args, use_container_width=True, **kwargs)


FIELD_LABELS = {
    "price": "Price", "brand": "Brand", "rating": "Rating", "stock_qty": "In stock",
    "weight_g": "Weight (g)",
}


def answer_badge(answer_type: str) -> None:
    icon, label, color = ANSWER_BADGES.get(answer_type, ("💬", answer_type, "gray"))
    st.markdown(f":{color}[**{icon} {label}**]")


def stock_badge(in_stock: bool, qty: int) -> str:
    if not in_stock or qty <= 0:
        return ":red[Out of stock]"
    if qty <= 3:
        return f":orange[Only {qty} left]"
    return f":green[In stock ({qty})]"


def product_card(product: Dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown(f"**{product['name']}**")
        st.caption(f"`{product['product_id']}` · {product.get('brand') or 'Brand not listed'}"
                   f" · {product.get('category') or ''}")
        price_col, rating_col = st.columns(2)
        price_col.markdown(f"### {api_client.format_inr(product.get('price'))}")
        rating = product.get("rating")
        rating_col.markdown(f"⭐ **{rating}**/5" if rating is not None else "⭐ not rated")
        st.markdown(stock_badge(product.get("in_stock", False), product.get("stock_qty") or 0))
        if product.get("reason"):
            st.markdown(f"💡 _{product['reason']}_")


def product_grid(products: List[Dict[str, Any]]) -> None:
    if not products:
        return
    columns = st.columns(min(len(products), 3))
    for i, product in enumerate(products):
        with columns[i % len(columns)]:
            product_card(product)


def comparison_table(comparison: Dict[str, Any]) -> None:
    rows = comparison.get("rows") or []
    if not rows:
        return
    data = {}
    for row in rows:
        values = dict(row.get("values") or {})
        if isinstance(values.get("price"), (int, float)):
            values["price"] = api_client.format_inr(values["price"])
        data[f"{row['name']} ({row['product_id']})"] = {
            FIELD_LABELS.get(f, f.replace("_", " ").capitalize()): str(values.get(f, "not listed"))
            for f in comparison.get("fields", [])
        }
    table(data)


def suggestion_chips(suggestions: List[str], key_prefix: str, on_click: Callable[[str], None]) -> None:
    if not suggestions:
        return
    columns = st.columns(len(suggestions))
    for i, text in enumerate(suggestions):
        with columns[i]:
            stretch(st.button, text, key=f"{key_prefix}-chip-{i}", on_click=on_click, args=(text,))


def trace_viewer(trace_id: str, key_prefix: str) -> None:
    with st.expander(f"🔎 Trace `{trace_id}` - how this answer was made"):
        if st.button("Load trace", key=f"{key_prefix}-trace"):
            try:
                st.session_state[f"trace-{trace_id}"] = api_client.trace(trace_id)
            except api_client.ApiError as exc:
                st.error(str(exc))
        trace = st.session_state.get(f"trace-{trace_id}")
        if not trace:
            st.caption("Shows the intent, retrieved products, filters, tools used and the final answer.")
            return
        intent = trace.get("intent") or {}
        st.markdown(f"**Intent:** `{intent.get('intent')}` via **{intent.get('source')}** · "
                    f"**LLM mode:** `{trace.get('llm_mode')}` · **Latency:** {trace.get('latency_ms')} ms")
        tools = trace.get("tools") or []
        if tools:
            st.markdown("**Tools run (in order)**")
            table([{"tool": t["name"], "result": t["output_summary"],
                    "ms": t["duration_ms"]} for t in tools], hide_index=True)
        retrieved = trace.get("retrieved") or []
        if retrieved:
            st.markdown("**Retrieved candidates (before filters)**")
            table(retrieved, hide_index=True)
        with st.popover("Full JSON"):
            st.json(trace)


def render_assistant_message(message: Dict[str, Any], index: int, on_chip: Callable[[str], None],
                             is_latest: bool) -> None:
    """Draw one assistant reply from the stored API response."""
    if message.get("error"):
        st.error(message["error"])
        return
    response = message["response"]
    answer_badge(response["answer_type"])
    st.markdown(response["answer"])
    follow_up = response.get("follow_up_question")
    if follow_up and follow_up != response["answer"]:
        st.info(follow_up)
    product_grid(response.get("products") or [])
    if response.get("comparison"):
        comparison_table(response["comparison"])
    if is_latest:   # chips only on the newest answer, so old ones can't be re-clicked out of context
        suggestion_chips(response.get("suggested_questions") or [], f"m{index}", on_chip)
    st.caption(f"trace `{response['trace_id']}` · {message.get('elapsed_ms', 0):.0f} ms · "
               f"language: {response.get('language', 'en')}")
    trace_viewer(response["trace_id"], f"m{index}")
