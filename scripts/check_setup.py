"""
Setup checker - every member runs this once after installing.

    python -m scripts.check_setup            # full check (pings Groq if a key is set)
    python -m scripts.check_setup --no-llm   # skip the Groq call

Prints OK / WARN / FAIL for each item. Only FAIL items block you.
"""
import argparse
import importlib
import sys
import time

from app.config import BASE_DIR, reload_settings
from app.db.database import fts5_available, init_db, table_counts

RESULTS = []


def report(status: str, item: str, detail: str = "") -> None:
    RESULTS.append(status)
    print(f"[{status:4}] {item}" + (f"  ->  {detail}" if detail else ""))


def check_python() -> None:
    v = sys.version_info
    if v >= (3, 10):
        report("OK", "Python version", f"{v.major}.{v.minor}.{v.micro}")
    else:
        report("FAIL", "Python version", f"{v.major}.{v.minor} found, need 3.10+")


def check_imports() -> None:
    packages = {
        "fastapi": "fastapi", "uvicorn": "uvicorn", "pydantic": "pydantic",
        "dotenv": "python-dotenv", "groq": "groq", "numpy": "numpy",
        "pandas": "pandas", "streamlit": "streamlit", "requests": "requests",
        "pytest": "pytest", "httpx": "httpx",
    }
    optional = {"faiss": "faiss-cpu", "sentence_transformers": "sentence-transformers"}
    for module, pip_name in packages.items():
        try:
            importlib.import_module(module)
            report("OK", f"import {module}")
        except ImportError:
            report("FAIL", f"import {module}", f"pip install {pip_name}")
    for module, pip_name in optional.items():
        try:
            importlib.import_module(module)
            report("OK", f"import {module}")
        except ImportError:
            report("WARN", f"import {module}",
                   f"pip install {pip_name}  (or set USE_VECTOR_SEARCH=false for keyword-only)")


def check_env(settings) -> None:
    if (BASE_DIR / ".env").exists():
        report("OK", ".env file found")
    else:
        report("WARN", ".env file missing", "copy .env.example to .env")
    report("OK", "LLM mode", settings.llm_mode)


def check_database(settings) -> None:
    if fts5_available():
        report("OK", "SQLite FTS5 available")
    else:
        report("FAIL", "SQLite FTS5 missing", "upgrade Python / SQLite")
    try:
        init_db()
        report("OK", "Database schema", f"{settings.db_path} -> {table_counts()}")
    except Exception as exc:  # noqa: BLE001 - show any setup error
        report("FAIL", "Database schema", str(exc))


def check_embedding_cache(settings) -> None:
    if not settings.use_vector_search:
        report("OK", "Vector search disabled", "keyword-only mode")
        return
    cache = settings.model_cache_dir
    has_weights = cache.exists() and (any(cache.rglob("*.safetensors")) or any(cache.rglob("*.bin")))
    if has_weights:
        report("OK", "Embedding model cached", str(cache))
    else:
        report("WARN", "Embedding model not downloaded", "run: python -m scripts.download_model")


def check_groq(settings) -> None:
    key_ok = settings.groq_api_key and not settings.groq_api_key.startswith("your_")
    if not key_ok:
        report("WARN", "Groq key not set", "mock mode will be used")
        return
    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key, timeout=settings.llm_timeout_seconds)
        model_ids = sorted(m.id for m in client.models.list().data)
        print("        Models available to your key:")
        for mid in model_ids:
            print(f"          - {mid}")
        if settings.groq_model not in model_ids:
            report("FAIL", f"GROQ_MODEL '{settings.groq_model}' not available",
                   "pick one from the list above and update .env")
            return
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            max_tokens=50,
            temperature=0,
        )
        ms = (time.perf_counter() - start) * 1000
        text = (resp.choices[0].message.content or "").strip()
        report("OK", f"Groq call to {settings.groq_model}", f"'{text[:30]}' in {ms:.0f} ms")
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "Groq call failed", f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="skip the Groq API call")
    args = parser.parse_args()

    settings = reload_settings()
    print("=== ShopSense AI setup check ===")
    check_python()
    check_imports()
    check_env(settings)
    check_database(settings)
    check_embedding_cache(settings)
    if not args.no_llm:
        check_groq(settings)

    fails = RESULTS.count("FAIL")
    warns = RESULTS.count("WARN")
    print(f"\nSummary: {RESULTS.count('OK')} OK, {warns} WARN, {fails} FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
