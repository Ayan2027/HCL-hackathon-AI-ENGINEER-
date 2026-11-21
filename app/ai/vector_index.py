"""
Semantic search index (Member 1) - FAISS + Sentence Transformers.

* Each product becomes one vector built from name + brand + category +
  attributes + description + review summary (instruction-like text removed).
* Cosine similarity = inner product of normalised vectors (IndexFlatIP).
* IndexIDMap2 lets us replace one product's vector, so POST /ingest updates
  search results immediately without a restart (R11).
* If FAISS or the model is unavailable, the index is "disabled" and search
  returns [] - retrieval then runs on keyword search alone.
"""
import hashlib
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from app.ai.safety import sanitize_catalog_text
from app.config import settings
from app.tools.normalize import attributes_to_text

logger = logging.getLogger("shopsense.vector")


def product_to_text(product: Dict[str, Any]) -> str:
    parts = [
        product.get("name"), product.get("brand"), product.get("category"), product.get("subcategory"),
        attributes_to_text(product.get("attributes") or {}), product.get("best_for"),
        sanitize_catalog_text(product.get("description")), sanitize_catalog_text(product.get("review_summary")),
    ]
    return ". ".join(str(p) for p in parts if p)


def catalogue_fingerprint(products: List[Dict[str, Any]]) -> str:
    """Changes whenever any product is added or updated -> saved index is rebuilt."""
    raw = "|".join(f"{p['product_id']}:{p.get('updated_at')}" for p in sorted(products, key=lambda x: x["product_id"]))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _numeric_id(product_id: str) -> int:
    return int(product_id.split("-")[1])          # "P-1001" -> 1001


class VectorIndex:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model = None
        self._index = None
        self.enabled = False
        self.error: Optional[str] = None

    # ---- setup ----
    def _load_model(self) -> bool:
        if self._model is not None:
            return True
        if not settings.use_vector_search:
            self.error = "USE_VECTOR_SEARCH=false"
            return False
        try:
            import faiss  # noqa: F401
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.embedding_model, cache_folder=str(settings.model_cache_dir))
            return True
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            logger.warning("Vector search disabled: %s", self.error)
            return False

    def _encode(self, texts: List[str]):
        import numpy as np

        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")

    def build(self, products: List[Dict[str, Any]]) -> int:
        """(Re)build the whole index from all products. Returns number of vectors."""
        with self._lock:
            if not self._load_model():
                self.enabled = False
                return 0
            import faiss
            import numpy as np

            dim = self._model.get_sentence_embedding_dimension()
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
            if products:
                vectors = self._encode([product_to_text(p) for p in products])
                ids = np.array([_numeric_id(p["product_id"]) for p in products], dtype="int64")
                index.add_with_ids(vectors, ids)
            self._index = index
            self.enabled = True
            self.error = None
            self._save(catalogue_fingerprint(products))
            return index.ntotal

    def load_or_build(self, products: List[Dict[str, Any]]) -> int:
        """Load the saved index if it matches the catalogue, otherwise rebuild."""
        with self._lock:
            if not self._load_model():
                self.enabled = False
                return 0
            import faiss

            index_file = settings.index_dir / "products.faiss"
            meta_file = settings.index_dir / "meta.json"
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if (index_file.exists() and meta.get("model") == settings.embedding_model
                        and meta.get("fingerprint") == catalogue_fingerprint(products)):
                    self._index = faiss.read_index(str(index_file))
                    self.enabled = True
                    logger.info("Loaded FAISS index with %d vectors", self._index.ntotal)
                    return self._index.ntotal
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            return self.build(products)

    def _save(self, fingerprint: str) -> None:
        try:
            import faiss

            settings.index_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(settings.index_dir / "products.faiss"))
            (settings.index_dir / "meta.json").write_text(
                json.dumps({"model": settings.embedding_model, "fingerprint": fingerprint}), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - saving is an optimisation only
            logger.warning("Could not save FAISS index: %s", exc)

    # ---- live updates (R11) ----
    def upsert(self, products: List[Dict[str, Any]]) -> bool:
        with self._lock:
            if not self.enabled or self._index is None or not products:
                return False
            import numpy as np

            ids = np.array([_numeric_id(p["product_id"]) for p in products], dtype="int64")
            self._index.remove_ids(ids)
            self._index.add_with_ids(self._encode([product_to_text(p) for p in products]), ids)
            self._save("updated-live")   # forces a clean rebuild on next start
            return True

    # ---- search ----
    def search(self, text: str, k: int = 30) -> List[Tuple[str, float]]:
        """Returns [(product_id, cosine_similarity)] best first."""
        with self._lock:
            if not self.enabled or self._index is None or self._index.ntotal == 0 or not text.strip():
                return []
            scores, ids = self._index.search(self._encode([text]), min(k, self._index.ntotal))
        results = []
        for score, num in zip(scores[0], ids[0]):
            if num >= 0:
                results.append((f"P-{int(num):04d}", float(score)))
        return results

    @property
    def size(self) -> int:
        return int(self._index.ntotal) if self.enabled and self._index is not None else 0


_vector_index = VectorIndex()


def get_vector_index() -> VectorIndex:
    return _vector_index
