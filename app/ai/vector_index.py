"""
Semantic search index (Member 1) - FAISS + Sentence Transformers.

* ONE vector per product (no chunking). The text is the product's own
  `embedding_text` from the catalogue; if that is empty, it is built only from
  verified fields of the same record (name, brand, category, subcategory,
  attributes, description). Price, stock and rating are never embedded - they
  are volatile and are filtered exactly in code.
* Instruction-like sentences are removed before embedding (catalogue text is data).
* Cosine similarity = inner product of normalised vectors (IndexFlatIP).
* Vector positions map to product IDs through id_map.json (saved next to the
  index), so any product ID format works and the mapping is reproducible.
* meta.json records the model, catalogue fingerprint, dimension, count and
  library versions; a saved index is reused only if all of them still match.
* If FAISS or the model is unavailable, the index is "disabled" and search
  returns [] - retrieval then runs on keyword search alone.
"""
import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.ai.safety import sanitize_catalog_text
from app.config import settings
from app.tools.normalize import attributes_to_text

logger = logging.getLogger("shopsense.vector")

INDEX_FILE = "products.faiss"
ID_MAP_FILE = "id_map.json"
META_FILE = "meta.json"
TEXT_VERSION = "embedding_text-v1"     # bump when product_to_text() changes


def product_to_text(product: Dict[str, Any]) -> str:
    """The text embedded for one product."""
    text = (product.get("embedding_text") or "").strip()
    if text:
        return sanitize_catalog_text(text) or ""
    parts = [
        product.get("name"), product.get("brand"), product.get("category"), product.get("subcategory"),
        attributes_to_text(product.get("attributes") or {}), product.get("best_for"),
        sanitize_catalog_text(product.get("description")), sanitize_catalog_text(product.get("review_summary")),
    ]
    return ". ".join(str(p) for p in parts if p)


def catalogue_fingerprint(products: List[Dict[str, Any]]) -> str:
    """Changes whenever the set of products or any embedded text changes -> index is rebuilt."""
    digest = hashlib.md5(TEXT_VERSION.encode("utf-8"))
    for p in sorted(products, key=lambda x: x["product_id"]):
        digest.update(f"{p['product_id']}\x00{product_to_text(p)}\x01".encode("utf-8"))
    return digest.hexdigest()


def _library_versions() -> Dict[str, str]:
    versions = {}
    for name in ("faiss", "sentence_transformers", "torch"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # noqa: BLE001
            versions[name] = "unavailable"
    return versions


class VectorIndex:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model = None
        self._index = None
        self._ids: List[str] = []              # vector id (int) -> product_id
        self._pos: Dict[str, int] = {}         # product_id -> vector id
        self.enabled = False
        self.error: Optional[str] = None
        self.meta: Dict[str, Any] = {}

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

    def _set_ids(self, ids: List[str]) -> None:
        self._ids = list(ids)
        self._pos = {pid: i for i, pid in enumerate(self._ids)}

    def build(self, products: List[Dict[str, Any]]) -> int:
        """(Re)build the whole index from the given (active) products. Returns number of vectors."""
        with self._lock:
            if not self._load_model():
                self.enabled = False
                return 0
            import faiss
            import numpy as np

            get_dim = (getattr(self._model, "get_embedding_dimension", None)
                       or self._model.get_sentence_embedding_dimension)
            dim = get_dim()
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
            ids = [p["product_id"] for p in products]
            if products:
                vectors = self._encode([product_to_text(p) for p in products])
                index.add_with_ids(vectors, np.arange(len(ids), dtype="int64"))
            self._index = index
            self._set_ids(ids)
            self.enabled = True
            self.error = None
            self._save(catalogue_fingerprint(products), dim)
            return index.ntotal

    def load_or_build(self, products: List[Dict[str, Any]]) -> int:
        """Load the saved index if it matches the catalogue and model, otherwise rebuild."""
        with self._lock:
            if not self._load_model():
                self.enabled = False
                return 0
            import faiss

            folder = settings.index_dir
            try:
                meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
                id_map = json.loads((folder / ID_MAP_FILE).read_text(encoding="utf-8"))
                if (meta.get("model") == settings.embedding_model
                        and meta.get("fingerprint") == catalogue_fingerprint(products)
                        and meta.get("count") == len(id_map) == len(products)):
                    index = faiss.read_index(str(folder / INDEX_FILE))
                    if index.ntotal == len(id_map):
                        self._index = index
                        self._set_ids(id_map)
                        self.meta = meta
                        self.enabled = True
                        logger.info("Loaded FAISS index with %d vectors", index.ntotal)
                        return index.ntotal
            except (FileNotFoundError, json.JSONDecodeError, RuntimeError):
                pass
            logger.info("FAISS index missing or stale - rebuilding from %d products", len(products))
            return self.build(products)

    def _save(self, fingerprint: str, dim: int) -> None:
        try:
            import faiss

            settings.index_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(settings.index_dir / INDEX_FILE))
            (settings.index_dir / ID_MAP_FILE).write_text(json.dumps(self._ids), encoding="utf-8")
            self.meta = {
                "model": settings.embedding_model, "fingerprint": fingerprint, "dim": dim,
                "count": len(self._ids), "text_source": TEXT_VERSION, "index_type": "IndexIDMap2(IndexFlatIP)",
                "normalized": True, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "libraries": _library_versions(),
            }
            (settings.index_dir / META_FILE).write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - saving is an optimisation only
            logger.warning("Could not save FAISS index: %s", exc)

    # ---- live updates (R11) ----
    def upsert(self, products: List[Dict[str, Any]]) -> bool:
        with self._lock:
            if not self.enabled or self._index is None or not products:
                return False
            import numpy as np

            new_ids = []
            for p in products:
                if p["product_id"] not in self._pos:
                    self._pos[p["product_id"]] = len(self._ids)
                    self._ids.append(p["product_id"])
                new_ids.append(self._pos[p["product_id"]])
            ids = np.array(new_ids, dtype="int64")
            self._index.remove_ids(ids)
            self._index.add_with_ids(self._encode([product_to_text(p) for p in products]), ids)
            # Fingerprint "updated-live" forces a clean rebuild from SQLite on the next start
            self._save("updated-live", self._index.d)
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
                if 0 <= num < len(self._ids):
                    results.append((self._ids[int(num)], float(score)))
        return results

    @property
    def size(self) -> int:
        return int(self._index.ntotal) if self.enabled and self._index is not None else 0


_vector_index = VectorIndex()


def get_vector_index() -> VectorIndex:
    return _vector_index
