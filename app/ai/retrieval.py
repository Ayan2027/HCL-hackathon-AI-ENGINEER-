"""
Hybrid retrieval (Member 1) - R3 "Search only the catalogue".

keyword (SQLite FTS5, BM25)   catches exact words: "Nike", "IPX7", "flat feet"
semantic (FAISS embeddings)   catches meaning: "gift for a trekker", Hindi text

score = w * semantic + (1 - w) * keyword_normalised      (w = SEMANTIC_WEIGHT, default 0.5)

A relevance gate drops weak matches, so an unrelated query returns nothing
(-> not_found) instead of random products. Every result is a real row from
the products table - the LLM is never asked to produce product IDs.
"""
from typing import Any, Dict, List, Tuple

from app.ai.vector_index import get_vector_index
from app.config import settings
from app.db import repository as repo
from app.schemas import Constraints

SEMANTIC_MIN = 0.35   # cosine similarity below this (and no keyword hit) = not relevant


def build_query_text(search_query_en: str, constraints: Constraints) -> str:
    parts = [search_query_en] + constraints.keywords + constraints.brands
    if constraints.occasion:
        parts.append(constraints.occasion)
    if constraints.color:
        parts.append(constraints.color)
    seen, words = set(), []
    for part in parts:
        for word in str(part or "").split():
            if word.lower() not in seen:
                seen.add(word.lower())
                words.append(word)
    return " ".join(words)


def hybrid_retrieve(
    query_text: str, constraints: Constraints, k: int = 0
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Returns (candidates, debug). Each candidate = product dict + retrieval scores."""
    k = k or settings.retrieval_k
    keyword_hits = repo.keyword_search(query_text, k) if query_text.strip() else []
    vector_index = get_vector_index()
    semantic_hits = vector_index.search(query_text, k) if vector_index.enabled else []

    weight = settings.semantic_weight if vector_index.enabled else 0.0
    max_kw = max((s for _, s in keyword_hits), default=0.0) or 1.0
    scores: Dict[str, Dict[str, float]] = {}
    for pid, score in keyword_hits:
        scores.setdefault(pid, {"keyword": 0.0, "semantic": 0.0})["keyword"] = score / max_kw
    for pid, score in semantic_hits:
        scores.setdefault(pid, {"keyword": 0.0, "semantic": 0.0})["semantic"] = max(0.0, score)

    gated = {
        pid: s for pid, s in scores.items()
        if s["keyword"] > 0 or s["semantic"] >= SEMANTIC_MIN
    }
    for s in gated.values():
        s["score"] = round(weight * s["semantic"] + (1 - weight) * s["keyword"], 4)

    source = "hybrid" if weight else "keyword"
    # Fallback: only a category was understood (no product words), e.g. "kuch kitchen ka saamaan".
    # With product words ("laptop") and no match, we return nothing -> honest not_found.
    if not gated and constraints.category and not constraints.keywords:
        for product in repo.get_products_by_category(constraints.category, limit=k):
            gated[product["product_id"]] = {"keyword": 0.0, "semantic": 0.0, "score": 0.05}
        source = "category_fallback"

    ordered = sorted(gated.items(), key=lambda item: item[1]["score"], reverse=True)
    products = {p["product_id"]: p for p in repo.get_products([pid for pid, _ in ordered])}
    candidates = []
    for pid, s in ordered:
        if pid in products:
            candidates.append({**products[pid], "_retrieval": {**s, "source": source}})

    debug = {
        "query": query_text,
        "mode": source,
        "semantic_weight": weight,
        "keyword_hits": len(keyword_hits),
        "semantic_hits": len(semantic_hits),
        "after_relevance_gate": len(candidates),
        "vector_error": vector_index.error,
    }
    return candidates, debug
