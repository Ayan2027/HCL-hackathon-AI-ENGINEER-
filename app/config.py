"""
Central configuration for ShopSense AI.

Every module reads settings from here - never call os.getenv() elsewhere.
Values come from the .env file in the project root (see .env.example).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = folder that contains "app/"
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Shared constants (frozen contract - all members use these exact values)
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Electronics",
    "Footwear",
    "Apparel",
    "Sports & Outdoors",
    "Home & Kitchen",
    "Beauty & Personal Care",
]

ANSWER_TYPES = [
    "recommendation",
    "comparison",
    "clarification",
    "product_fact",
    "not_found",
    "refusal",
]

LANGUAGES = ["en", "hi", "hinglish"]

PRODUCT_ID_PATTERN = r"^P-\d{4}$"      # e.g. P-1001
CUSTOMER_ID_PATTERN = r"^C-\d{3}$"     # e.g. C-001
ORDER_ID_PATTERN = r"^O-\d{5}$"        # e.g. O-00001


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else BASE_DIR / path


class Settings:
    """All runtime settings. Re-create with reload_settings() in tests."""

    def __init__(self) -> None:
        # LLM
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        self.mock_llm: bool = _bool("MOCK_LLM", True)
        self.llm_timeout_seconds: float = _float("LLM_TIMEOUT_SECONDS", 20.0)
        self.llm_temperature: float = _float("LLM_TEMPERATURE", 0.2)
        self.llm_max_tokens: int = _int("LLM_MAX_TOKENS", 800)

        # Storage
        self.db_path: Path = _path("DB_PATH", "data/shopsense.db")
        self.index_dir: Path = _path("INDEX_DIR", "data/index")
        self.sample_dir: Path = BASE_DIR / "data" / "sample"
        self.raw_dir: Path = BASE_DIR / "data" / "raw"

        # Retrieval
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ).strip()
        self.model_cache_dir: Path = _path("MODEL_CACHE_DIR", "models")
        self.use_vector_search: bool = _bool("USE_VECTOR_SEARCH", True)
        self.retrieval_k: int = _int("RETRIEVAL_K", 30)
        self.top_k: int = _int("TOP_K", 3)
        self.semantic_weight: float = _float("SEMANTIC_WEIGHT", 0.5)

        # Conversation
        self.max_clarifications: int = _int("MAX_CLARIFICATIONS", 2)
        self.history_turns: int = _int("HISTORY_TURNS", 6)

        # Frontend
        self.api_url: str = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

        self.log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    @property
    def use_mock_llm(self) -> bool:
        """Mock mode if requested OR if no real key is configured (demo never crashes)."""
        key_missing = (not self.groq_api_key) or self.groq_api_key.startswith("your_")
        return self.mock_llm or key_missing

    @property
    def llm_mode(self) -> str:
        return "mock" if self.use_mock_llm else f"groq:{self.groq_model}"


settings = Settings()


def reload_settings() -> Settings:
    """Re-read environment variables (used by tests). Updates the shared object in place."""
    load_dotenv(BASE_DIR / ".env", override=False)
    fresh = Settings()
    settings.__dict__.update(fresh.__dict__)
    return settings
