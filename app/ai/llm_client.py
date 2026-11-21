"""
Groq LLM client (Member 1).

One public function: chat_json(system, user) -> dict.
- Uses Groq JSON mode so the reply is machine-readable.
- Raises LLMUnavailable in mock mode or on any failure, so the caller can
  fall back to the rule-based path. The app therefore never crashes because
  of the LLM (no key, no internet, rate limit, bad JSON).
"""
import json
import logging
import re
import threading
import time
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger("shopsense.llm")

_client = None
_client_lock = threading.Lock()
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class LLMUnavailable(Exception):
    """Mock mode, missing key, network error, or unusable output."""


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            from groq import Groq  # imported lazily so mock mode works without the package

            _client = Groq(api_key=settings.groq_api_key, timeout=settings.llm_timeout_seconds, max_retries=1)
    return _client


def _parse_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)        # model wrapped JSON in extra text
        if not match:
            raise LLMUnavailable("no JSON in model output")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMUnavailable("JSON is not an object")
    return data


def _extra_body() -> Optional[Dict[str, Any]]:
    # gpt-oss models "think" before answering; low effort keeps replies fast.
    if settings.groq_model.lower().startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    return None


def chat_json(system: str, user: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Call Groq and return the parsed JSON object. Raises LLMUnavailable on any problem."""
    if settings.use_mock_llm:
        raise LLMUnavailable("mock mode")

    kwargs: Dict[str, Any] = dict(
        model=settings.groq_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=settings.llm_temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
    )
    extra = _extra_body()
    if extra:
        kwargs["extra_body"] = extra

    start = time.perf_counter()
    try:
        client = _get_client()
        try:
            response = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
        except Exception as first_error:  # noqa: BLE001
            # Some models reject JSON mode or fail its validation: retry once in plain mode.
            logger.warning("JSON mode failed (%s); retrying without it", first_error)
            response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
    except LLMUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
    finally:
        logger.info("Groq call took %.0f ms", (time.perf_counter() - start) * 1000)

    return _parse_json(content)
