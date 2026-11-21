"""
Data clean-up helpers (Member 2).

The catalogue is deliberately messy: weights in g and kg, missing brands,
sizes as numbers or letters. These functions turn raw values into exact,
comparable values with plain Python - no LLM involved.
"""
import re
from typing import Any, Dict, List, Optional

_WEIGHT_RE = re.compile(r"^\s*([\d.,]+)\s*(kg|kgs|kilogram|kilograms|g|gm|gms|gram|grams|mg|lb|lbs)\s*$", re.I)
_TO_GRAMS = {"kg": 1000, "kgs": 1000, "kilogram": 1000, "kilograms": 1000,
             "g": 1, "gm": 1, "gms": 1, "gram": 1, "grams": 1,
             "mg": 0.001, "lb": 453.592, "lbs": 453.592}


def normalize_weight(raw: Optional[str]) -> Optional[float]:
    """'350 g' -> 350.0, '0.95 kg' -> 950.0, '200g' -> 200.0.
    Volumes ('100 ml') and unknown text return None (not a weight)."""
    if raw is None:
        return None
    match = _WEIGHT_RE.match(str(raw))
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    return round(value * _TO_GRAMS[match.group(2).lower()], 1)


def clean_text(value: Any) -> Optional[str]:
    """Strip whitespace; empty strings become None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def attributes_to_text(attributes: Dict[str, Any]) -> str:
    """{'color': 'Blue', 'sizes_available': [8, 9]} -> 'color Blue sizes available 8 9' (for search)."""
    parts: List[str] = []
    for key, value in (attributes or {}).items():
        if isinstance(value, (list, tuple)):
            value = " ".join(str(v) for v in value)
        elif isinstance(value, bool):
            value = "yes" if value else "no"
        parts.append(f"{str(key).replace('_', ' ')} {value}")
    return " ".join(parts)


def size_matches(requested: Optional[str], available: Any) -> Optional[bool]:
    """True/False if the product lists sizes; None if it has no size information."""
    if requested is None:
        return None
    if not isinstance(available, (list, tuple)) or not available:
        return None
    want = str(requested).strip().upper()
    return any(str(s).strip().upper() == want for s in available)


def color_matches(requested: Optional[str], product_color: Any) -> Optional[bool]:
    if requested is None:
        return None
    if not product_color:
        return None
    return requested.strip().lower() in str(product_color).strip().lower()
