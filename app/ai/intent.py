"""
Intent extraction (Member 1) - R1 "Understand ordinary language".

Two paths that produce the SAME Intent object:
  * LLM path  : Groq reads the message and returns JSON (best understanding).
  * Mock path : rule-based parser (MOCK_LLM=true, or whenever the LLM fails).

Exact values (budget numbers, sizes, product IDs, positions like "first two")
are always extracted by regex in code and override the LLM - the LLM is used
for meaning, code is used for numbers.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.ai import llm_client
from app.ai.language import TRANSLATIONS, to_english_keywords, tokenize
from app.ai.prompts import INTENT_SYSTEM, build_intent_user_prompt
from app.config import CATEGORIES
from app.schemas import Constraints, Intent, SessionState

logger = logging.getLogger("shopsense.intent")

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
CATEGORY_WORDS: Dict[str, List[str]] = {
    "Electronics": ["speaker", "speakers", "headphone", "headphones", "earphone", "earphones", "earbuds",
                    "buds", "phone", "phones", "mobile", "smartphone", "laptop", "laptops", "tablet",
                    "smartwatch", "smart watch", "watch", "charger", "power bank", "powerbank", "tv",
                    "television", "camera", "mouse", "keyboard", "bluetooth", "neckband", "monitor"],
    "Footwear": ["shoe", "shoes", "sneaker", "sneakers", "sandal", "sandals", "slipper", "slippers",
                 "boot", "boots", "footwear", "flip flops", "heels", "loafers"],
    "Apparel": ["shirt", "shirts", "t-shirt", "tshirt", "t shirt", "tee", "jeans", "jacket", "jackets",
                "kurta", "dress", "hoodie", "trouser", "trousers", "saree", "clothes", "sweater",
                "shorts", "track pants", "trackpants", "top"],
    "Sports & Outdoors": ["backpack", "daypack", "rucksack", "trekking", "hiking", "tent", "yoga",
                          "dumbbell", "dumbbells", "cycle", "bicycle", "cricket", "football", "racket",
                          "racquet", "sleeping bag", "trekking pole", "poles", "camping", "gym",
                          "skipping rope", "badminton"],
    "Home & Kitchen": ["cooker", "pressure cooker", "pan", "frying pan", "fry pan", "kadai", "kadhai",
                       "mixer", "grinder", "kettle", "bedsheet", "pillow", "lamp", "cookware", "tawa",
                       "lunch box", "tiffin", "knife", "utensil", "utensils", "kitchen", "non-stick",
                       "nonstick", "cooking"],
    "Beauty & Personal Care": ["face wash", "facewash", "shampoo", "serum", "moisturizer", "moisturiser",
                               "sunscreen", "cream", "lipstick", "perfume", "trimmer", "soap", "lotion",
                               "conditioner", "deodorant", "hair oil", "skincare", "skin care"],
}
# Things this store does not sell -> out_of_scope (not_found answer)
UNSOLD_WORDS = ["car", "cars", "motorcycle", "motorbike", "scooter", "flight", "flights", "air ticket",
                "train ticket", "hotel", "medicine", "medicines", "grocery", "groceries", "vegetables",
                "insurance", "loan", "property", "apartment", "plot", "real estate", "sofa", "refrigerator",
                "fridge", "washing machine", "air conditioner", "gold coin", "petrol", "tractor"]
COLORS = ["black", "white", "blue", "red", "green", "grey", "gray", "yellow", "pink", "purple", "orange",
          "brown", "navy", "silver", "beige", "maroon"]
RECIPIENTS = {"father": "father", "dad": "father", "papa": "father", "mother": "mother", "mom": "mother",
              "mummy": "mother", "brother": "brother", "sister": "sister", "friend": "friend",
              "wife": "wife", "husband": "husband", "son": "son", "daughter": "daughter",
              "girlfriend": "girlfriend", "boyfriend": "boyfriend"}
FILLER_WORDS = {
    "show", "me", "find", "need", "want", "looking", "look", "for", "a", "an", "the", "some", "any", "i",
    "please", "can", "you", "do", "sell", "have", "get", "buy", "my", "with", "and", "or", "of", "to", "in",
    "on", "is", "are", "it", "this", "that", "these", "those", "one", "ones", "only", "just", "under",
    "below", "above", "within", "upto", "up", "less", "more", "than", "around", "about", "budget", "rs",
    "inr", "rupees", "k", "price", "cheap", "cheaper", "cheapest", "affordable", "costly", "expensive",
    "premium", "best", "good", "nice", "top", "rated", "rating", "size", "colour", "color", "brand", "brands",
    "something", "thing", "things", "item", "items", "product", "products", "option", "options", "other",
    "which", "what", "who", "likes", "like", "loves", "love", "enjoys", "gift", "present", "birthday",
    "not", "no", "except", "without", "but", "also", "too", "very", "really", "much", "many", "same",
    "show", "dikhao", "first", "second", "third", "two", "three", "both", "compare", "vs", "versus",
    "available", "stock", "there", "here", "would", "should", "could", "will", "be", "was", "under",
    "lakh", "thousand", "hazar", "now", "today", "please", "plz", "pls", "hey", "hi", "hello", "uk", "us",
    "number", "no", "between", "at", "least", "max", "maximum", "minimum", "min", "from", "by", "your",
    "who", "he", "she", "they", "him", "her", "them", "his", "their", "loves", "go", "goes", "going",
    "ke", "ka", "ki", "liye", "wala", "wali", "wale", "ek", "koi", "sirf", "aur", "se", "kam", "tak",
    "waist", "inch", "inches",
}

_NUM = r"(?:₹|rs\.?|inr)?\s*(\d+(?:[.,]\d+)*)\s*(k|thousand|hazar|hazaar|हज़ार|हजार|lakh|lac|लाख)?"
_MAX_BEFORE = re.compile(r"(?:under|below|less than|within|upto|up to|max(?:imum)?|not more than|budget(?: is| of)?|around|about|<=?)\s*" + _NUM, re.I)
_CUR_WORD = r"\s*(?:rs\.?|rupees|rupaye|rupay|inr|रुपये|रुपए|रुपया)?"
_MAX_AFTER = re.compile(_NUM + _CUR_WORD + r"\s*(?:se kam|se neeche|tak|ke andar|ke neeche|or less|or below|max|से कम|तक|के अंदर|के नीचे)", re.I)
_MIN_BEFORE = re.compile(r"(?:above|over|more than|at least|minimum|min|>=?)\s*" + _NUM, re.I)
_MIN_AFTER = re.compile(_NUM + _CUR_WORD + r"\s*(?:se zyada|se jyada|se upar|or more|plus|\+|से ज़्यादा|से ज्यादा|से ऊपर)", re.I)
_BETWEEN = re.compile(r"between\s*" + _NUM + r"\s*(?:and|&|to|-)\s*" + _NUM, re.I)
_RANGE = re.compile(r"(?:₹|rs\.?|inr)\s*(\d+(?:[.,]\d+)*)\s*(k)?\s*(?:-|to)\s*(?:₹|rs\.?|inr)?\s*(\d+(?:[.,]\d+)*)\s*(k)?", re.I)
_CURRENCY_ONLY = re.compile(r"(?:₹|rs\.?\s|inr\s)\s*(\d+(?:[.,]\d+)*)\s*(k)?|(\d+(?:[.,]\d+)*)\s*(k)?\s*(?:rs|rupees|inr|rupaye|रुपये|रुपए)\b", re.I)

_SIZE_PATTERNS = [
    re.compile(r"\bsize\s*(?:no\.?|number|of)?\s*([0-9]{1,2}|xxxl|xxl|xl|xs|s|m|l)\b", re.I),
    re.compile(r"(?:साइज़|साइज)\s*([0-9]{1,2}|xxxl|xxl|xl|xs|s|m|l)", re.I),
    re.compile(r"\b(?:uk|us|euro?)\s*([0-9]{1,2})\b", re.I),
    re.compile(r"\b([0-9]{1,2})\s*(?:number|no\.?)\s*(?:ka|ke|ki|wala|wale)\b", re.I),
    re.compile(r"\b(xxxl|xxl|xl|xs)\b", re.I),
    re.compile(r"\b(2[6-9]|3[0-9]|4[0-4])\s*(?:inch(?:es)?\s*)?waist\b", re.I),
    re.compile(r"\bwaist\s*(2[6-9]|3[0-9]|4[0-4])\b", re.I),
]
_PRODUCT_ID = re.compile(r"\bP-\d{4}\b", re.I)
_MIN_RATING = re.compile(r"(?:rated|rating)\s*(?:above|over|at least|more than)?\s*(\d(?:\.\d)?)|(\d(?:\.\d)?)\s*(?:\+\s*)?stars?", re.I)

_ORDINALS = {
    1: r"first|1st|pehla|pehle|pahla|पहला|पहले",
    2: r"second|2nd|doosra|dusra|doosre|दूसरा|दूसरे",
    3: r"third|3rd|teesra|tisra|तीसरा|तीसरे",
}
_FIRST_N = re.compile(r"\b(?:first|top|pehle|pahle)\s+(two|2|three|3|do|teen)\b|पहले\s+(दो|तीन)", re.I)
_BOTH = re.compile(r"\b(both|dono|these two|in dono|those two)\b|दोनों", re.I)
_OPTION_N = re.compile(r"\b(?:option|number|no\.?|#)\s*([1-9])\b", re.I)
_LAST = re.compile(r"\b(last one|last)\b|आखिरी", re.I)

_MY_ORDERS = re.compile(
    r"\b(my|mere|meri|mera)\b.{0,25}\b(orders?|purchases?|order history|kharid\w*|bought)\b"
    r"|\bwhat (did|have) i (buy|bought|order|ordered|purchase|purchased)\b"
    r"|\bmaine\b.{0,15}\b(kharida|order kiya|liya)\b|मैंने.{0,15}(खरीदा|ऑर्डर)|मेरे ऑर्डर", re.I)
_COMPARE = re.compile(r"\b(compare|comparison|vs\.?|versus|difference between|differences|tulna|"
                      r"which is better|which one is better|kaun\s*sa behtar|kaunsa accha|kaun sa accha)\b|तुलना", re.I)
_STOCK = re.compile(r"\b(in stock|available|availability|stock|milega|milegi|mil jayega|hai kya|left)\b"
                    r"|उपलब्ध|स्टॉक|मिलेगा", re.I)
_REFERENCE = re.compile(r"\b(it|this|that|this one|that one|the \w+ one|ye|yeh|wo|woh|isme|iska|uska|iski)\b"
                        r"|यह|वह|इसका|इसकी", re.I)
_FACT_WORDS = re.compile(
    r"\b(price|cost|kitna|kitne|kitni|how much|what is|what's|battery|warranty|material|weight|wajan|vajan|"
    r"sizes?|colou?rs?|rating|reviews?|waterproof|capacity|specs?|specifications?|details|tell me about|"
    r"kya hai|kaisa hai|daam|keemat)\b|कीमत|दाम|कितना|कितने", re.I)
_YES_NO_QUESTION = re.compile(r"^\s*(is|are|does|can|will|kya)\s+(the|this|that|it|ye|yeh)\b", re.I)
_CHITCHAT = re.compile(r"^\s*(hi+|hello|hey|namaste|namaskar|thanks|thank you|thank u|thx|ok|okay|bye|"
                       r"shukriya|dhanyavad|good morning|good evening|नमस्ते|धन्यवाद)[\s!.]*$", re.I)
_SORT_ASC = re.compile(r"\b(cheap|cheaper|cheapest|budget|affordable|low price|lowest price|inexpensive|"
                       r"sasta|sasti|saste|kam daam)\b|सस्ता|सस्ती|सस्ते", re.I)
_SORT_RATING = re.compile(r"\b(best rated|top rated|highest rated|best reviews|best reviewed)\b", re.I)
_SORT_DESC = re.compile(r"\b(premium|expensive|costly|high end|mehnga|mehenga)\b|महंगा", re.I)
_NEGATION = re.compile(r"\b(not|no|except|without|other than|nahi|mat|na)\b|नहीं", re.I)
_ATTRIBUTE_MAP = [
    (r"price|cost|kitna|kitne|kitni|how much|daam|keemat|कीमत|दाम|कितना", "price"),
    (r"battery", "battery_hours"),
    (r"warranty", "warranty"),
    (r"material|fabric", "material"),
    (r"weight|wajan|vajan|heavy|light", "weight"),
    (r"sizes?", "sizes_available"),
    (r"colou?rs?", "color"),
    (r"rating|reviews?", "rating"),
    (r"waterproof|water resistant", "waterproof"),
    (r"capacity|litre|liter", "capacity_l"),
    (r"stock|available|availability", "stock_qty"),
]


# ---------------------------------------------------------------------------
# Small extractors (used by both paths)
# ---------------------------------------------------------------------------
def _to_number(value: str, unit: Optional[str]) -> Optional[float]:
    try:
        number = float(value.replace(",", ""))
    except (TypeError, ValueError):
        return None
    unit = (unit or "").lower()
    if unit in {"k", "thousand", "hazar", "hazaar", "हज़ार", "हजार"}:
        number *= 1000
    elif unit in {"lakh", "lac", "लाख"}:
        number *= 100000
    return number


def extract_budget(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (budget_min, budget_max) found in the text, using exact regex rules."""
    t = text.lower()
    m = _BETWEEN.search(t)
    if m:
        lo, hi = _to_number(m.group(1), m.group(2)), _to_number(m.group(3), m.group(4))
        if lo is not None and hi is not None:
            return min(lo, hi), max(lo, hi)
    m = _RANGE.search(t)
    if m:
        lo, hi = _to_number(m.group(1), m.group(2)), _to_number(m.group(3), m.group(4))
        if lo is not None and hi is not None:
            return min(lo, hi), max(lo, hi)
    budget_min = budget_max = None
    for regex in (_MAX_BEFORE, _MAX_AFTER):
        m = regex.search(t)
        if m:
            budget_max = _to_number(m.group(1), m.group(2))
            break
    for regex in (_MIN_BEFORE, _MIN_AFTER):
        m = regex.search(t)
        if m:
            budget_min = _to_number(m.group(1), m.group(2))
            break
    if budget_max is None and budget_min is None:
        m = _CURRENCY_ONLY.search(t)
        if m:
            budget_max = _to_number(m.group(1) or m.group(3), m.group(2) or m.group(4))
    # Ignore tiny numbers that are clearly not prices (e.g. "under 5" stars)
    if budget_max is not None and budget_max < 50:
        budget_max = None
    if budget_min is not None and budget_min < 50:
        budget_min = None
    return budget_min, budget_max


def extract_size(text: str) -> Optional[str]:
    for pattern in _SIZE_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).upper()
    return None


def extract_color(text: str) -> Optional[str]:
    words = set(tokenize(text))
    for word in words:
        translated = TRANSLATIONS.get(word)
        if translated in COLORS:
            return translated
    for color in COLORS:
        if color in words:
            return "grey" if color == "gray" else color
    return None


def extract_positions(text: str) -> List[int]:
    """'compare the first two' -> [1, 2]; 'the second one' -> [2]; 'option 3' -> [3]."""
    t = text.lower()
    m = _FIRST_N.search(t)
    if m:
        word = (m.group(1) or m.group(2) or "").lower()
        n = 3 if word in {"three", "3", "teen", "तीन"} else 2
        return list(range(1, n + 1))
    if _BOTH.search(t):
        return [1, 2]
    positions: List[int] = []
    for pos, pattern in _ORDINALS.items():
        if re.search(rf"(?:\b|^)({pattern})(?:\b|$)", t):
            positions.append(pos)
    for m in _OPTION_N.finditer(t):
        positions.append(int(m.group(1)))
    if _LAST.search(t):
        positions.append(-1)
    return sorted(set(positions), key=lambda p: (p < 0, p))


def extract_product_ids(text: str) -> List[str]:
    return list(dict.fromkeys(pid.upper() for pid in _PRODUCT_ID.findall(text)))


def _phrase_in(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", text) is not None


def detect_category(text_en: str) -> Tuple[Optional[str], List[str]]:
    """Return (category, matched words). Picks the category with most word hits."""
    best, best_hits = None, []
    for category, words in CATEGORY_WORDS.items():
        hits = [w for w in words if _phrase_in(text_en, w)]
        if len(hits) > len(best_hits):
            best, best_hits = category, hits
    return best, best_hits


# Brands that are also everyday words: only a brand when written like one ("Noise", not "noise cancellation")
COMMON_WORD_BRANDS = {"noise", "plum", "engage", "w", "victor", "spaces", "core", "go", "kent", "mi"}


def detect_brands(text: str, known_brands: List[str]) -> Tuple[List[str], List[str]]:
    """Return (include_brands, exclude_brands) by matching brand names from the database."""
    include: List[str] = []
    exclude: List[str] = []
    lowered = text.lower()
    for brand in sorted(known_brands, key=len, reverse=True):
        m = re.search(rf"(?<![a-z0-9]){re.escape(brand.lower())}(?![a-z0-9])", lowered)
        if not m:
            continue
        if brand.lower() in COMMON_WORD_BRANDS:
            original = text[m.start():m.end()]
            following = lowered[m.end():m.end() + 12].strip()
            if not original[:1].isupper() or following.startswith(("cancel", "free", "less")):
                continue
        before = lowered[max(0, m.start() - 20):m.start()]
        after = lowered[m.end():m.end() + 12]
        negated = _NEGATION.search(before) or re.search(r"^\s*(nahi|mat|नहीं)", after)
        (exclude if negated else include).append(brand)
    return include, exclude


def detect_attribute(text: str) -> Optional[str]:
    for pattern, attribute in _ATTRIBUTE_MAP:
        if re.search(rf"\b({pattern})\b", text, re.I) or (not pattern.isascii() and re.search(pattern, text)):
            return attribute
    return None


def detect_occasion(words_en: List[str]) -> Optional[str]:
    is_gift = any(w in {"gift", "present", "birthday", "anniversary"} for w in words_en)
    recipient = next((RECIPIENTS[w] for w in words_en if w in RECIPIENTS), None)
    if is_gift and recipient:
        return f"gift for {recipient}"
    if is_gift:
        return "gift"
    if recipient and "for" in words_en:
        return f"for {recipient}"
    return None


# ---------------------------------------------------------------------------
# Mock (rule-based) parser
# ---------------------------------------------------------------------------
def mock_parse(message: str, state: SessionState, language: str, known_brands: List[str]) -> Intent:
    text = message.strip()
    lowered = text.lower()
    english_words = to_english_keywords(text)
    text_en = " ".join(english_words)                     # Hindi words translated
    combined = f"{lowered} {text_en}"

    category, category_hits = detect_category(combined)
    budget_min, budget_max = extract_budget(text)
    brands, exclude_brands = detect_brands(text, known_brands)
    size = extract_size(text)
    color = extract_color(text)
    positions = extract_positions(text)
    product_ids = extract_product_ids(text)
    has_previous = bool(state.last_result_ids)
    brand_words = {w for b in known_brands for w in tokenize(b)}

    min_rating = None
    m = _MIN_RATING.search(lowered)
    if m:
        value = float(m.group(1) or m.group(2))
        if 1 <= value <= 5:
            min_rating = value

    sort_by = "relevance"
    if _SORT_RATING.search(combined):
        sort_by = "rating"
    elif _SORT_ASC.search(combined):
        sort_by = "price_asc"
    elif _SORT_DESC.search(combined):
        sort_by = "price_desc"

    content = [w for w in english_words
               if w not in FILLER_WORDS and w not in brand_words and w not in COLORS
               and not w.isdigit() and len(w) > 1 and w not in RECIPIENTS]
    content = list(dict.fromkeys(content))

    refers_to_previous = bool(positions) or (has_previous and bool(_REFERENCE.search(text)))
    is_reference_only = not category and not content

    # ---- intent decision (order matters) ----
    if _MY_ORDERS.search(text):
        intent = "my_orders"
    elif _CHITCHAT.match(text):
        intent = "chitchat"
    elif _COMPARE.search(combined):
        intent = "compare"
        refers_to_previous = refers_to_previous or (has_previous and not product_ids and not brands)
    elif _STOCK.search(text) and (product_ids or refers_to_previous or (has_previous and is_reference_only)
                                  or (has_previous and color and not category)):
        intent = "stock_check"
        refers_to_previous = refers_to_previous or not product_ids
    elif (product_ids or (refers_to_previous and not category)) and _FACT_WORDS.search(text):
        intent = "product_fact"
    elif (_YES_NO_QUESTION.search(text) and (category or content) and text.rstrip().endswith("?")
          and not _COMPARE.search(combined)):
        intent = "product_fact"
    elif any(_phrase_in(combined, w) for w in UNSOLD_WORDS) and not category:
        intent = "out_of_scope"
    elif has_previous and is_reference_only and (
        budget_max or budget_min or brands or exclude_brands or size or color or min_rating
        or sort_by != "relevance" or re.search(r"\b(more|other|aur|any brand|koi bhi)\b|और|कोई भी", lowered)
    ):
        intent = "refine"
        refers_to_previous = True
    else:
        intent = "search"

    keywords = content[:8]
    occasion = detect_occasion(english_words)
    search_query = " ".join(content) or " ".join(category_hits) or text_en

    constraints = Constraints(
        category=category, budget_min=budget_min, budget_max=budget_max, brands=brands,
        exclude_brands=exclude_brands, size=size, color=color, keywords=keywords,
        occasion=occasion, min_rating=min_rating, sort_by=sort_by,
    )
    return Intent(
        intent=intent, constraints=constraints, search_query_en=search_query.strip(),
        refers_to_previous=refers_to_previous, referenced_positions=positions,
        product_ids_mentioned=product_ids, attribute_asked=detect_attribute(text),
        language=language, confidence=0.6,
    )


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------
def _normalise_category(value: Any) -> Optional[str]:
    if not value:
        return None
    for category in CATEGORIES:
        if str(value).strip().lower() == category.lower():
            return category
    return None


def _normalise_brands(values: Any, known_brands: List[str]) -> List[str]:
    if not isinstance(values, list):
        return []
    lookup = {b.lower(): b for b in known_brands}
    out = []
    for v in values:
        if v and str(v).strip():
            out.append(lookup.get(str(v).strip().lower(), str(v).strip()))
    return list(dict.fromkeys(out))


def _intent_from_llm(data: Dict[str, Any], rules: Intent, language: str, known_brands: List[str]) -> Intent:
    """Validate LLM JSON, then let code-extracted exact values win."""
    raw = dict(data.get("constraints") or {})
    raw["category"] = _normalise_category(raw.get("category"))
    raw["brands"] = _normalise_brands(raw.get("brands"), known_brands)
    raw["exclude_brands"] = _normalise_brands(raw.get("exclude_brands"), known_brands)
    raw["keywords"] = [str(k).lower() for k in (raw.get("keywords") or []) if k][:8]
    if raw.get("sort_by") not in {"relevance", "price_asc", "price_desc", "rating"}:
        raw["sort_by"] = "relevance"
    for key in ("budget_min", "budget_max", "min_rating"):
        try:
            raw[key] = float(raw[key]) if raw.get(key) not in (None, "") else None
        except (TypeError, ValueError):
            raw[key] = None
    raw.pop("in_stock_only", None)
    constraints = Constraints(**{k: v for k, v in raw.items() if k in Constraints.model_fields})

    rc = rules.constraints
    # Code wins for exact values it found
    if rc.budget_min is not None or rc.budget_max is not None:
        constraints.budget_min, constraints.budget_max = rc.budget_min, rc.budget_max
    if rc.size:
        constraints.size = rc.size
    if rc.brands or rc.exclude_brands:
        constraints.brands, constraints.exclude_brands = rc.brands, rc.exclude_brands
    if constraints.budget_min and constraints.budget_max and constraints.budget_min > constraints.budget_max:
        constraints.budget_min, constraints.budget_max = constraints.budget_max, constraints.budget_min
    if not constraints.category:
        constraints.category = rc.category

    intent_name = data.get("intent")
    valid = {"search", "refine", "compare", "product_fact", "stock_check", "my_orders", "chitchat", "out_of_scope"}
    if intent_name not in valid:
        intent_name = rules.intent
    # Privacy-sensitive routing stays rule-based: only the regex may grant "my_orders"
    if intent_name == "my_orders" and rules.intent != "my_orders":
        intent_name = "search"

    positions = rules.referenced_positions or [int(p) for p in (data.get("referenced_positions") or [])
                                               if str(p).lstrip("-").isdigit()]
    product_ids = list(dict.fromkeys(rules.product_ids_mentioned + [
        str(p).upper() for p in (data.get("product_ids_mentioned") or []) if _PRODUCT_ID.fullmatch(str(p))]))

    return Intent(
        intent=intent_name,
        constraints=constraints,
        search_query_en=str(data.get("search_query_en") or rules.search_query_en)[:200],
        refers_to_previous=bool(data.get("refers_to_previous")) or rules.refers_to_previous,
        referenced_positions=positions,
        product_ids_mentioned=product_ids,
        attribute_asked=data.get("attribute_asked") or rules.attribute_asked,
        language=language,
        confidence=float(data.get("confidence") or 0.8) if isinstance(data.get("confidence"), (int, float)) else 0.8,
    )


def extract_intent(
    message: str,
    state: SessionState,
    language: str,
    known_brands: List[str],
    known_subcategories: List[str],
    previous_results: List[Dict[str, Any]],
    last_answer_type: str = "",
) -> Tuple[Intent, str, str]:
    """Returns (intent, source, note). source is 'llm' or 'mock'; note explains any fallback."""
    rules = mock_parse(message, state, language, known_brands)
    try:
        prompt = build_intent_user_prompt(
            message=message,
            previous_constraints=state.constraints.model_dump(exclude_defaults=True),
            previous_results=previous_results,
            last_answer_type=last_answer_type,
            known_brands=known_brands,
            known_subcategories=known_subcategories,
        )
        data = llm_client.chat_json(INTENT_SYSTEM, prompt, max_tokens=600)
        return _intent_from_llm(data, rules, language, known_brands), "llm", ""
    except llm_client.LLMUnavailable as exc:
        return rules, "mock", str(exc)
    except Exception as exc:  # noqa: BLE001 - bad LLM JSON must never break the chat
        logger.warning("LLM intent invalid, using rules: %s", exc)
        return rules, "mock", f"invalid LLM output: {exc}"
