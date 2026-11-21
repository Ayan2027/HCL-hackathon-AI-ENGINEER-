"""
Safety guards (Member 1) - R8 (content is data, not orders) and R9 (privacy).

These are plain-code checks that run BEFORE any LLM call, so a clever prompt
cannot talk its way past them. A second layer lives in prompts.py: product
text is wrapped in <catalog_data> tags and the LLM is told it is data.
"""
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

_INJECTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(instruction|instructions|rules|prompt|above|previous|earlier)\b",
    r"\bdisregard\b.{0,40}\b(instruction|instructions|rules|prompt|above|previous)\b",
    r"\bforget\b.{0,30}\b(instruction|instructions|rules|everything|previous)\b",
    r"\b(system|developer)\s*(prompt|message|instruction|mode)\b",
    r"^\s*system\s*:",
    r"\byou are now\b",
    r"\bact as\b.{0,30}\b(admin|developer|dan|unrestricted|system)\b",
    r"\b(jailbreak|do anything now)\b",
    r"\breveal\b.{0,30}\b(prompt|instructions|system)\b",
    r"\boverride\b.{0,30}\b(rules|instructions|safety|policy)\b",
    r"\brecommend this (item|product)\b.{0,40}\b(every|all|always)\b",
    r"\bfor every (query|question|search)\b",
    r"\binstructions?\b.{0,20}\b(ignore|bhool|chhod)\s*(karo|do|jao)?\b",
    r"\b(nirdesh|rules)\b.{0,20}\b(bhool|ignore|chhod)\b",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _INJECTION_PATTERNS]

# Asking about orders / data of someone else
_OTHER_PERSON_DATA = [
    r"\bwhat (did|has|have)\s+(?!i\b|me\b|my\b)([a-z]+)\s+(buy|bought|order|ordered|purchase|purchased)",
    r"\b(orders?|purchases?|purchase history|order history|address|phone number|email)\s+of\s+(?!me\b|mine\b|my\b)",
    r"\b(?!my\b|mine\b|me\b)([a-z]+)'s\s+(orders?|purchases?|history|address|phone|email|profile|cart)",
    r"\b(all|every|other)\s+(customers?|shoppers?|users?)\b",
    r"\b(list|show|give)\b.{0,20}\b(customers?|shoppers?|users?)\b",
    r"\b([a-z]+)\s+ne\s+kya\s+(kharida|order kiya|liya)",
    r"\b(dusre|doosre|kisi aur)\b.{0,20}\b(customer|shopper|user|grahak)\b",
    r"\b(customer|shopper|user)\s*(id)?\s*c-\d{3}\b",
]
_OTHER_PERSON_RE = [re.compile(p, re.IGNORECASE) for p in _OTHER_PERSON_DATA]
_DATA_WORDS = re.compile(
    r"\b(buy|bought|order|ordered|orders|purchase|purchased|history|address|phone|email|kharida|liya|details|cart)\b",
    re.IGNORECASE,
)
_CUSTOMER_ID_RE = re.compile(r"\bC-\d{3}\b", re.IGNORECASE)


@dataclass
class SafetyVerdict:
    kind: str = "ok"          # "ok" | "injection" | "privacy"
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return self.kind != "ok"


def looks_like_injection(text: str) -> bool:
    return any(p.search(text or "") for p in _INJECTION_RE)


def asks_for_other_shopper(
    message: str, current_customer_id: str, customers: Iterable[Tuple[str, str]]
) -> Optional[str]:
    """Return a reason string if the message asks about another shopper's data, else None."""
    text = message or ""
    # 1. Another customer ID typed in the message
    for cid in _CUSTOMER_ID_RE.findall(text):
        if cid.upper() != current_customer_id.upper():
            return f"mentions another customer id {cid.upper()}"
    # 2. Another shopper's name together with a data word (buy, orders, address ...)
    if _DATA_WORDS.search(text):
        lowered = text.lower()
        for cid, name in customers:
            if cid.upper() == current_customer_id.upper() or not name:
                continue
            first = name.split()[0].lower()
            if re.search(rf"\b{re.escape(first)}\b", lowered):
                return f"asks about shopper '{first}'"
    # 3. Generic patterns ("what did Priya buy", "orders of ...", "all customers")
    for pattern in _OTHER_PERSON_RE:
        match = pattern.search(text)
        if match:
            return f"pattern: {match.group(0)[:40]}"
    return None


def check_message(
    message: str, current_customer_id: str, customers: Iterable[Tuple[str, str]]
) -> SafetyVerdict:
    """Run both guards on the shopper's message. Privacy is checked first (R9 > R8)."""
    privacy_reason = asks_for_other_shopper(message, current_customer_id, customers)
    if privacy_reason:
        return SafetyVerdict("privacy", privacy_reason)
    if looks_like_injection(message):
        return SafetyVerdict("injection", "instruction-like text in the shopper message")
    return SafetyVerdict()


def sanitize_catalog_text(text: Optional[str]) -> Optional[str]:
    """Remove instruction-like sentences from product descriptions/reviews before the LLM sees them."""
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    clean = [s for s in sentences if not looks_like_injection(s)]
    if len(clean) != len(sentences):
        clean.append("[instruction-like text removed]")
    return " ".join(clean)
