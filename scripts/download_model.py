"""
Download the embedding model ONCE, during setup (it is ~470 MB).

    python -m scripts.download_model

If this fails (slow Wi-Fi), set USE_VECTOR_SEARCH=false in .env.
The assistant then runs on keyword search only and still works.
"""
import sys
import time

from app.config import reload_settings


def main() -> int:
    settings = reload_settings()
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{settings.embedding_model}' into {settings.model_cache_dir} ...")
    start = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.embedding_model, cache_folder=str(settings.model_cache_dir))
        vectors = model.encode(
            ["running shoes for flat feet", "सस्ता ब्लूटूथ स्पीकर", "sasta bluetooth speaker"],
            normalize_embeddings=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print("Fallback: set USE_VECTOR_SEARCH=false in .env (keyword-only search).")
        return 1

    secs = time.perf_counter() - start
    print(f"OK - embedding dimension {vectors.shape[1]}, took {secs:.1f}s")
    # Quick sanity check: Hindi and Hinglish speaker queries should be similar
    similarity = float(vectors[1] @ vectors[2])
    print(f"Similarity('सस्ता ब्लूटूथ स्पीकर', 'sasta bluetooth speaker') = {similarity:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
