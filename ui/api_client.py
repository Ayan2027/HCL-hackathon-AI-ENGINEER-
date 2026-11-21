"""
HTTP client for the ShopSense API (Member 3).

The UI never touches the database or the AI code directly - it only calls
the same public endpoints the judges use. Every call has a timeout and turns
network problems into a friendly ApiError instead of a crash.
"""
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:  # read API_URL from the project .env when running locally
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover
    pass

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
CHAT_TIMEOUT = 60      # seconds; first Groq call + model warm-up can be slow
SHORT_TIMEOUT = 8


class ApiError(Exception):
    """Readable error for the UI."""


def _request(method: str, path: str, timeout: float, **kwargs) -> Tuple[Any, float]:
    url = f"{API_URL}{path}"
    start = time.perf_counter()
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.ConnectionError as exc:
        raise ApiError(f"Cannot reach the API at {API_URL}. Is it running? "
                       "Start it with: uvicorn app.main:app --port 8000") from exc
    except requests.Timeout as exc:
        raise ApiError(f"The API took longer than {timeout:.0f}s to answer. Please try again.") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000

    if response.status_code == 404:
        raise ApiError("Not found.")
    if response.status_code == 422:
        detail = response.json().get("detail")
        raise ApiError(f"The API rejected the input: {detail}")
    if response.status_code >= 400:
        raise ApiError(f"API error {response.status_code}: {response.text[:200]}")
    return response.json(), elapsed_ms


def health() -> Dict[str, Any]:
    data, _ = _request("GET", "/health", SHORT_TIMEOUT)
    return data


def customers() -> List[Dict[str, Any]]:
    data, _ = _request("GET", "/customers", SHORT_TIMEOUT)
    return data


def chat(session_id: str, customer_id: str, message: str) -> Tuple[Dict[str, Any], float]:
    """Returns (ChatResponse JSON, round-trip time in ms)."""
    return _request("POST", "/chat", CHAT_TIMEOUT,
                    json={"session_id": session_id, "customer_id": customer_id, "message": message})


def product(product_id: str) -> Dict[str, Any]:
    data, _ = _request("GET", f"/products/{product_id.strip().upper()}", SHORT_TIMEOUT)
    return data


def trace(trace_id: str) -> Dict[str, Any]:
    data, _ = _request("GET", f"/trace/{trace_id}", SHORT_TIMEOUT)
    return data


def ingest(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    data, _ = _request("POST", "/ingest", CHAT_TIMEOUT, json={"products": products})
    return data


def format_inr(amount: Optional[float]) -> str:
    """2499 -> '₹2,499'; 150000 -> '₹1,50,000' (Indian grouping)."""
    if amount is None:
        return "Price not listed"
    s = str(int(round(amount)))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups + [tail])
    return f"₹{s}"
