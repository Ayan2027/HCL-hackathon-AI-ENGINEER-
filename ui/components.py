"""
Reusable Streamlit pieces (Member 3): page styling, status chips, answer badges,
product cards, comparison table, follow-up chips and the trace viewer.

Everything shown here comes from the API response - the UI adds no facts.
Catalogue text is untrusted data, so every value placed inside HTML is escaped.
"""
import html
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

import requests
import streamlit as st

import api_client

try:                     # pandas is only used for tables; keep the UI usable without it
    import pandas as pd
except Exception:        # noqa: BLE001 - e.g. blocked numpy DLLs on locked-down Windows
    pd = None

ANSWER_BADGES = {
    "recommendation": ("🛍️", "Recommendation", "green"),
    "comparison": ("⚖️", "Comparison", "blue"),
    "clarification": ("❓", "Quick question", "orange"),
    "product_fact": ("📋", "Product info", "violet"),
    "not_found": ("🔍", "Not in catalogue", "gray"),
    "refusal": ("🛡️", "Can't help with that", "red"),
}

FIELD_LABELS = {
    "price": "Price", "retail_price": "MRP", "brand": "Brand", "category": "Category",
    "subcategory": "Subcategory", "rating": "Rating", "stock_qty": "Stock (simulated demo inventory)",
    "weight_g": "Weight (g)",
}
NOT_AVAILABLE = "Not available in catalogue"
SIMULATED_CAPTION = "Stock levels are simulated demo inventory, not live retailer stock."

# Neutral, semi-transparent colours so the same CSS works in light and dark themes
CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.ss-hero h1 {font-size: 2rem; margin: 0 0 .25rem 0;}
.ss-hero p {opacity: .75; margin: 0 0 .75rem 0;}
.ss-chips {display: flex; flex-wrap: wrap; gap: .4rem; margin: .25rem 0 1rem 0;}
.ss-chip {font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
          border: 1px solid rgba(128,128,128,.35); background: rgba(128,128,128,.08); white-space: nowrap;}
.ss-chip.ok {border-color: rgba(34,197,94,.55); background: rgba(34,197,94,.12);}
.ss-chip.warn {border-color: rgba(245,158,11,.6); background: rgba(245,158,11,.14);}
.ss-card {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: .85rem .95rem;
          background: rgba(128,128,128,.05); height: 100%; display: flex; flex-direction: column; gap: .35rem;}
.ss-card:hover {border-color: rgba(128,128,128,.55);}
.ss-img {height: 150px; display: flex; align-items: center; justify-content: center;
         background: #fff; border-radius: 10px; overflow: hidden; margin-bottom: .25rem;}
.ss-img img {max-height: 140px; max-width: 100%; object-fit: contain;}
.ss-meta {font-size: .74rem; opacity: .7; letter-spacing: .01em;}
.ss-meta code {font-size: .72rem;}
.ss-name {font-weight: 600; font-size: .95rem; line-height: 1.3;
          display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;}
.ss-price {font-size: 1.35rem; font-weight: 700;}
.ss-mrp {font-size: .8rem; opacity: .6; text-decoration: line-through; margin-left: .4rem; font-weight: 400;}
.ss-row {display: flex; align-items: center; justify-content: space-between; gap: .5rem; flex-wrap: wrap;}
.ss-pill {font-size: .72rem; padding: .12rem .5rem; border-radius: 999px; font-weight: 600;}
.ss-pill.in {background: rgba(34,197,94,.16); color: rgb(34,160,80);}
.ss-pill.low {background: rgba(245,158,11,.18); color: rgb(200,120,0);}
.ss-pill.out {background: rgba(239,68,68,.16); color: rgb(220,60,60);}
.ss-rating {font-size: .8rem; font-weight: 600;}
.ss-tags {display: flex; flex-wrap: wrap; gap: .25rem;}
.ss-tag {font-size: .7rem; padding: .1rem .45rem; border-radius: 6px; background: rgba(128,128,128,.12);}
.ss-reason {font-size: .82rem; opacity: .9; border-left: 3px solid rgba(99,102,241,.6); padding-left: .5rem;}
.ss-tile {border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: .6rem .75rem;
          background: rgba(128,128,128,.05); font-size: .85rem;}
.ss-tile b {display: block; font-size: .95rem; margin-bottom: .1rem;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def table(data, **kwargs) -> None:
    """Show tabular data with pandas when available, otherwise Streamlit's plain table."""
    if pd is not None:
        stretch(st.dataframe, pd.DataFrame(data), **kwargs)
    else:                # st.table takes no pandas-only options such as hide_index
        stretch(st.table, data)


def stretch(widget, *args, **kwargs):
    """Full-width widget on any Streamlit version (new: width="stretch", old: use_container_width)."""
    try:
        return widget(*args, width="stretch", **kwargs)
    except TypeError:
        return widget(*args, use_container_width=True, **kwargs)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def status_chips(status: Optional[Dict[str, Any]]) -> None:
    """Catalogue / search / LLM status, plus a banner when the API is stale or keyword-only."""
    if not status:
        return
    hybrid = status.get("retrieval_mode") == "hybrid" or (status.get("vector_index") or 0) > 0
    llm = "Mock LLM (offline)" if status.get("mock_llm") else f"LLM: {status.get('llm_mode', '').replace('groq:', 'Groq ')}"
    chips = [
        ("ok" if status.get("status") == "ok" else "warn", f"● {status.get('products', 0)} products"),
        ("ok" if hybrid else "warn",
         f"Hybrid search · {status.get('vector_index')} vectors" if hybrid else "Keyword-only search"),
        ("", llm),
    ]
    if status.get("inventory_simulated"):
        chips.append(("", "Simulated stock"))
    st.markdown('<div class="ss-chips">' + "".join(
        f'<span class="ss-chip {cls}">{esc(text)}</span>' for cls, text in chips) + "</div>",
        unsafe_allow_html=True)

    if "retrieval_mode" not in status:
        st.warning("The API on this address is running an **older version** of the backend "
                   "(it does not report its retrieval mode). Stop it and start it again from `.venv312`:  \n"
                   "`.\\.venv312\\Scripts\\python.exe -m uvicorn app.main:app --port 8000`")
    elif not hybrid:
        reason = status.get("vector_status") or "vector index not loaded"
        hint = ("set `USE_VECTOR_SEARCH=true` in `.env` and restart the API"
                if "USE_VECTOR_SEARCH" in reason else
                "run the API with Python 3.12 (`.venv312`) and build the index: `python -m scripts.build_index`")
        st.info(f"Semantic (vector) search is off - **{reason}**. Answers still work with keyword search. "
                f"To enable it, {hint}.")


def answer_badge(answer_type: str) -> None:
    icon, label, color = ANSWER_BADGES.get(answer_type, ("💬", answer_type, "gray"))
    st.markdown(f":{color}[**{icon} {label}**]")


def stock_badge(in_stock: bool, qty: int, simulated: bool = False) -> str:
    """Markdown stock label (kept for callers outside the HTML card)."""
    note = " · simulated" if simulated else ""
    if not in_stock or qty <= 0:
        return f":red[Out of stock{note}]"
    if qty <= 3:
        return f":orange[Only {qty} left{note}]"
    return f":green[In stock ({qty}){note}]"


def _stock_pill(product: Dict[str, Any]) -> str:
    qty = product.get("stock_qty") or 0
    note = " · simulated" if product.get("stock_is_simulated") else ""
    if not product.get("in_stock") or qty <= 0:
        return f'<span class="ss-pill out">Out of stock{note}</span>'
    if qty <= 3:
        return f'<span class="ss-pill low">Only {qty} left{note}</span>'
    return f'<span class="ss-pill in">In stock ({qty}){note}</span>'


# ---------------------------------------------------------------------------
# Images: shown only when the link really serves an image (no broken placeholders)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def image_is_reachable(url: str) -> bool:
    """True only if the URL answers with an image. Checked once per hour per URL."""
    try:
        response = requests.head(url, timeout=2, allow_redirects=True)
        if response.status_code == 405:                     # some hosts refuse HEAD
            response = requests.get(url, timeout=2, stream=True)
            response.close()
        return response.ok and response.headers.get("content-type", "").startswith("image/")
    except requests.RequestException:
        return False


def _valid_url(url: Any) -> str:
    url = str(url or "").strip()
    return url if url.lower().startswith(("http://", "https://")) else ""


def check_images(products: List[Dict[str, Any]]) -> Dict[str, bool]:
    """Check all card images in parallel (a dead host would otherwise add 2 s per card)."""
    urls = list(dict.fromkeys(u for u in (_valid_url(p.get("image_url")) for p in products) if u))
    if not urls:
        return {}
    with ThreadPoolExecutor(max_workers=min(8, len(urls))) as pool:
        return dict(zip(urls, pool.map(image_is_reachable, urls)))


def product_image(url: Any) -> None:
    url = _valid_url(url)
    if url and image_is_reachable(url):
        st.image(url, width=160)


# ---------------------------------------------------------------------------
# Product cards
# ---------------------------------------------------------------------------
def product_card_html(product: Dict[str, Any], image_ok: bool) -> str:
    url = _valid_url(product.get("image_url"))
    image = (f'<div class="ss-img"><img src="{esc(url)}" alt="" loading="lazy" referrerpolicy="no-referrer"></div>'
             if url and image_ok else "")
    leaf = str(product.get("subcategory") or "").split(">")[-1].strip()
    meta = [f"<code>{esc(product['product_id'])}</code>", esc(product.get("brand") or "Brand not listed")]
    if product.get("category"):
        meta.append(esc(product["category"] + (f" › {leaf}" if leaf else "")))

    price = product.get("price")
    mrp = product.get("retail_price")
    mrp_html = (f'<span class="ss-mrp">MRP {esc(api_client.format_inr(mrp))}</span>'
                if mrp and price is not None and mrp > price else "")
    rating = (f'<span class="ss-rating">⭐ {esc(product["rating"])}/5</span>'
              if product.get("rating") is not None else "")     # shown only when the catalogue has one

    attrs = product.get("attributes") or {}
    tags = "".join(f'<span class="ss-tag">{esc(k.replace("_", " ").capitalize())}: {esc(v)}</span>'
                   for k, v in list(attrs.items())[:5])
    reason = f'<div class="ss-reason">💡 {esc(product["reason"])}</div>' if product.get("reason") else ""

    return (
        f'<div class="ss-card">{image}'
        f'<div class="ss-meta">{" · ".join(meta)}</div>'
        f'<div class="ss-name" title="{esc(product["name"])}">{esc(product["name"])}</div>'
        f'<div class="ss-row"><span class="ss-price">{esc(api_client.format_inr(price))}{mrp_html}</span>{rating}</div>'
        f'<div>{_stock_pill(product)}</div>'
        f'{f"<div class=ss-tags>{tags}</div>" if tags else ""}{reason}</div>'
    )


def product_card(product: Dict[str, Any]) -> None:
    images = check_images([product])
    st.markdown(product_card_html(product, images.get(_valid_url(product.get("image_url")), False)),
                unsafe_allow_html=True)


def product_grid(products: List[Dict[str, Any]]) -> None:
    if not products:
        return
    images = check_images(products)
    columns = st.columns(min(len(products), 3))
    for i, product in enumerate(products):
        with columns[i % len(columns)]:
            ok = images.get(_valid_url(product.get("image_url")), False)
            st.markdown(product_card_html(product, ok), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Comparison, chips, trace
# ---------------------------------------------------------------------------
def comparison_table(comparison: Dict[str, Any]) -> None:
    rows = comparison.get("rows") or []
    if not rows:
        return
    data = {}
    for row in rows:
        values = dict(row.get("values") or {})
        for money in ("price", "retail_price"):
            if isinstance(values.get(money), (int, float)):
                values[money] = api_client.format_inr(values[money])
        data[f"{row['name'][:40]} ({row['product_id']})"] = {
            FIELD_LABELS.get(f, f.replace("_", " ").capitalize()): str(values.get(f, NOT_AVAILABLE))
            for f in comparison.get("fields", [])
        }
    st.markdown("**Side-by-side**")
    table(data)
    st.caption(f"“{NOT_AVAILABLE}” means the field is missing from our data - not zero or negative. "
               + SIMULATED_CAPTION)


def suggestion_chips(suggestions: List[str], key_prefix: str, on_click: Callable[[str], None]) -> None:
    if not suggestions:
        return
    columns = st.columns(len(suggestions))
    for i, text in enumerate(suggestions):
        with columns[i]:
            stretch(st.button, f"↪ {text}", key=f"{key_prefix}-chip-{i}", on_click=on_click, args=(text,))


def trace_viewer(trace_id: str, key_prefix: str) -> None:
    with st.expander(f"🔎 How this answer was made · trace `{trace_id}`"):
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
    products = response.get("products") or []
    product_grid(products)
    if response.get("comparison"):
        comparison_table(response["comparison"])
    elif any(p.get("stock_is_simulated") for p in products):
        st.caption(SIMULATED_CAPTION)
    if is_latest:   # chips only on the newest answer, so old ones can't be re-clicked out of context
        suggestion_chips(response.get("suggested_questions") or [], f"m{index}", on_chip)
    st.caption(f"{message.get('elapsed_ms', 0) / 1000:.1f} s · language: {response.get('language', 'en')}")
    trace_viewer(response["trace_id"], f"m{index}")
