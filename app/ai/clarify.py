"""
Clarifying questions (Member 1) - R2 "Ask when unsure".

Rules (all in code, easy to explain):
  * Ask only for a NEW search that is too vague to rank well.
  * At most settings.max_clarifications (2) questions in a row - then recommend anyway.
  * Never ask for a slot the shopper already gave or we already asked.
The slot choice works like a salesperson's "buying criteria" table.
"""
from typing import List, Optional, Tuple

from app.ai.language import clarify_question
from app.config import settings
from app.schemas import Constraints, Intent, SessionState

# Words that name a product family but say nothing about what the shopper needs
GENERIC_NOUNS = {
    "shoes", "shoe", "footwear", "clothes", "clothing", "dress", "gift", "something", "electronics",
    "laptop", "laptops", "phone", "phones", "mobile", "smartphone", "tv", "television", "camera",
    "watch", "smartwatch", "tablet", "kitchen", "beauty", "skincare", "accessories", "bag", "bags",
    "product", "item", "stuff", "present", "headphones", "earphones", "shirt", "shirts",
}

# Buying criteria: what a good salesperson asks for each product family
USE_EXAMPLES = {
    "laptop": " - study, gaming or office work",
    "laptops": " - study, gaming or office work",
    "phone": " - camera, gaming or long battery life",
    "mobile": " - camera, gaming or long battery life",
    "smartphone": " - camera, gaming or long battery life",
    "shoes": " - running, office or casual wear",
    "shoe": " - running, office or casual wear",
    "footwear": " - running, office or casual wear",
    "clothes": " - casual, office or a festival",
    "shirt": " - casual or office",
    "shirts": " - casual or office",
    "headphones": " - gym, travel or calls",
    "earphones": " - gym, travel or calls",
    "watch": " - fitness tracking or style",
    "smartwatch": " - fitness tracking or style",
    "bag": " - college, travel or trekking",
}


def _is_vague(intent: Intent, merged: Constraints) -> Tuple[bool, List[str]]:
    c = merged
    has_preference = any([
        c.budget_min, c.budget_max, c.brands, c.size, c.color, c.min_rating,
        c.sort_by != "relevance", intent.product_ids_mentioned,
    ])
    keywords = [k for k in c.keywords if k]
    specific = [k for k in keywords if k not in GENERIC_NOUNS]
    if not c.category and not keywords and not c.occasion:
        return True, ["product_type"]
    if c.occasion == "gift" and not c.category and not specific:
        return True, ["recipient"]
    if c.occasion and c.occasion.startswith(("gift for", "for ")) and not specific and not c.category:
        return True, ["product_type", "budget"]
    if not specific and not has_preference and not c.occasion:
        return True, ["budget", "use"]
    return False, []


def plan_clarification(
    intent: Intent, merged: Constraints, state: SessionState, language: str
) -> Optional[Tuple[str, List[str]]]:
    """Return (question, slots_asked) or None if we should search now."""
    if intent.intent != "search":
        return None
    if state.clarify_count >= settings.max_clarifications:
        return None                                   # R2: max two in a row
    vague, slots = _is_vague(intent, merged)
    if not vague:
        return None

    # Never re-ask a slot already asked or already answered
    remaining = [s for s in slots if s not in state.asked_slots]
    if "budget" in remaining and (merged.budget_max or merged.budget_min):
        remaining.remove("budget")
    if not remaining:
        return None
    if remaining[0] == "product_type" and len(remaining) > 1 and "product_type" in state.asked_slots:
        remaining = remaining[1:]

    noun = next((k for k in merged.keywords if k in USE_EXAMPLES), "")
    examples = USE_EXAMPLES.get(noun, "")
    if remaining == ["budget", "use"]:
        question = clarify_question("budget_use", language, examples=examples)
    elif remaining == ["use"]:
        question = clarify_question("use", language, examples=examples)
    else:
        question = clarify_question(remaining[0], language)
        remaining = remaining[:1]
    return question, remaining
