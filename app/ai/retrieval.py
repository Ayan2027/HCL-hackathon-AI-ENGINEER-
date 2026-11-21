"""
Hybrid retrieval (Member 1) - R3 "Search only the catalogue".

keyword (SQLite FTS5, BM25)   catches exact words: brand names, model numbers, "USB"
semantic (FAISS embeddings)   catches meaning: "gift for a coffee lover", Hindi text

Fusion = Reciprocal Rank Fusion (RRF), k = 60:
    RRF(d) = sum over lists r of 1 / (k + rank_r(d))
The score is divided by its maximum possible value (n_lists / (k + 1)), so the
top result of every list scores 1.0 and ranking bonuses stay on a 0-1 scale.

A relevance gate drops weak matches first (a keyword hit, or cosine >= SEMANTIC_MIN),
so an unrelated query returns nothing (-> not_found) instead of random products.
Results are deduplicated by product_id, and every result is a real row from the
ACTIVE catalogue - the LLM never produces product IDs. Hard filters (budget,
brand, stock ...) run afterwards in tools.apply_filters, so similarity can never
override them.
"""
from typing import Any, Dict, List, Tuple

from app.ai.vector_index import get_vector_index
from app.config import settings
from app.db import repository as repo
from app.schemas import Constraints

SEMANTIC_MIN = 0.35   # cosine similarity below this (and no keyword hit) = not relevant
RRF_K = 60            # conventional RRF constant


def rrf_fuse(ranked_lists: List[List[str]], k: int = RRF_K) -> Dict[str, float]:
    """Reciprocal Rank Fusion over ranked product-ID lists, normalised to 0..1. Deduplicates IDs."""
    lists = [list(dict.fromkeys(ids)) for ids in ranked_lists if ids]
    if not lists:
        return {}
    scores: Dict[str, float] = {}
    for ids in lists:
        for rank, pid in enumerate(ids, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    best_possible = len(lists) / (k + 1)
    return {pid: round(score / best_possible, 4) for pid, score in scores.items()}


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

    max_kw = max((s for _, s in keyword_hits), default=0.0) or 1.0
    scores: Dict[str, Dict[str, float]] = {}
    for pid, score in keyword_hits:
        scores.setdefault(pid, {"keyword": 0.0, "semantic": 0.0})["keyword"] = round(score / max_kw, 4)
    for pid, score in semantic_hits:
        scores.setdefault(pid, {"keyword": 0.0, "semantic": 0.0})["semantic"] = round(max(0.0, score), 4)

    # Relevance gate, then RRF over the gated keyword and semantic rankings
    keyword_ranked = [pid for pid, _ in keyword_hits]
    semantic_ranked = [pid for pid, s in semantic_hits if s >= SEMANTIC_MIN]
    fused = rrf_fuse([keyword_ranked, semantic_ranked])
    gated = {pid: {**scores[pid], "score": fused[pid]} for pid in fused}

    source = "hybrid" if vector_index.enabled else "keyword"
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
        "fusion": f"rrf(k={RRF_K})",
        "keyword_hits": len(keyword_hits),
        "semantic_hits": len(semantic_hits),
        "after_relevance_gate": len(candidates),
        "vector_error": vector_index.error,
    }
    return candidates, debug
