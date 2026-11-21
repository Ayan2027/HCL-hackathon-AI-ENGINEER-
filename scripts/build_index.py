"""
Build and verify the FAISS vector index from the ACTIVE catalogue in SQLite.

    python -m scripts.build_index            # rebuild, save, verify
    python -m scripts.build_index --check    # only check that FAISS + the model load

Run it after `python -m app.db.catalog_import`. It always rebuilds from the
current database, so an index made from an older catalogue is never reused.
Set USE_VECTOR_SEARCH=true in .env only after this command reports OK.
Exit code 0 = index built and verified.
"""
import argparse
import json
import sys

from app.config import settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and verify the FAISS product index")
    parser.add_argument("--check", action="store_true", help="only verify faiss/model imports")
    args = parser.parse_args()

    try:
        import faiss
        print(f"faiss {faiss.__version__} imported OK")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: faiss cannot be imported ({type(exc).__name__}: {exc}). "
              "Use a Python version with a faiss-cpu wheel (3.12 recommended) - see docs/SETUP.md.")
        return 1
    if args.check:
        return 0

    settings.use_vector_search = True            # this command's job is to build the index
    from app.ai.vector_index import get_vector_index
    from app.db import repository as repo
    from app.db.database import init_db

    init_db()
    products = repo.get_all_products()           # active catalogue only (archived rows excluded)
    if not products:
        print("FAIL: no active products in the database. Run python -m app.db.catalog_import first.")
        return 1
    index = get_vector_index()
    count = index.build(products)
    if not index.enabled:
        print(f"FAIL: could not build the index: {index.error}")
        return 1

    problems = []
    if count != len(products):
        problems.append(f"vector count {count} != active products {len(products)}")
    active_ids = {p["product_id"] for p in products}
    # Each product should find itself as the nearest neighbour of its own text
    sample = products[:: max(1, len(products) // 12)]
    from app.ai.vector_index import product_to_text
    self_hits = sum(1 for p in sample if index.search(product_to_text(p), 1)[:1]
                    and index.search(product_to_text(p), 1)[0][0] == p["product_id"])
    for query in ["wrist watch for men", "मोबाइल कवर", "coffee mug", "bedsheet"]:
        hits = index.search(query, 3)
        stale = [pid for pid, _ in hits if pid not in active_ids]
        if stale:
            problems.append(f"stale ids for '{query}': {stale}")
        print(f"  '{query}' -> {[(pid, round(score, 3)) for pid, score in hits]}")
    print(f"Self-retrieval: {self_hits}/{len(sample)} sampled products are their own nearest neighbour")
    if self_hits < len(sample) * 0.8:
        problems.append("self-retrieval below 80%")

    print(json.dumps(index.meta, indent=2))
    if problems:
        print("FAIL: " + "; ".join(problems))
        return 1
    print(f"OK: {count} vectors saved to {settings.index_dir}. You can now set USE_VECTOR_SEARCH=true.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
