"""
Shared Pydantic contracts for ShopSense AI.

FROZEN CONTRACT: all four members import these models. Do not rename fields
without telling the whole team - the API, UI, AI engine and eval scripts
all depend on them.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import CUSTOMER_ID_PATTERN, ORDER_ID_PATTERN, PRODUCT_ID_PATTERN

AnswerType = Literal[
    "recommendation", "comparison", "clarification", "product_fact", "not_found", "refusal"
]
Language = Literal["en", "hi", "hinglish"]
IntentName = Literal[
    "search",        # find products ("running shoes for flat feet")
    "refine",        # change previous results ("cheaper ones", "only Nike")
    "compare",       # compare 2+ products ("compare the first two")
    "product_fact",  # one product's price / attribute ("what is the battery life of P-1001?")
    "stock_check",   # availability ("is the blue one in size 9?")
    "my_orders",     # the CURRENT shopper's own order history
    "chitchat",      # greetings, thanks
    "out_of_scope",  # not about shopping in this store
]
SortBy = Literal["relevance", "price_asc", "price_desc", "rating"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
class ProductIn(BaseModel):
    """A complete product as stored in data/raw/products.json and the products table."""

    model_config = ConfigDict(extra="ignore")

    product_id: str = Field(..., pattern=PRODUCT_ID_PATTERN)
    name: str = Field(..., min_length=2, max_length=200)
    category: str
    subcategory: Optional[str] = None
    brand: Optional[str] = None                       # may be missing (messy data)
    price: Optional[float] = Field(None, ge=0)        # INR; may be missing
    stock_qty: int = Field(0, ge=0)
    rating: Optional[float] = Field(None, ge=0, le=5)
    rating_count: Optional[int] = Field(None, ge=0)
    attributes: Dict[str, Any] = Field(default_factory=dict)  # color, sizes_available, ...
    description: Optional[str] = None
    review_summary: Optional[str] = None
    reviews: List[str] = Field(default_factory=list)
    best_for: Optional[str] = None
    weight_raw: Optional[str] = None                  # "350 g" / "0.95 kg" (inconsistent units)
    language: Language = "en"                         # language of description/reviews


class ProductPatch(BaseModel):
    """Body item for POST /ingest. Only product_id is required.
    Existing products: given fields are updated, others kept.
    New products: name and category are also required (checked by the repository)."""

    model_config = ConfigDict(extra="ignore")

    product_id: str = Field(..., pattern=PRODUCT_ID_PATTERN)
    name: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = Field(None, ge=0)
    stock_qty: Optional[int] = Field(None, ge=0)
    rating: Optional[float] = Field(None, ge=0, le=5)
    rating_count: Optional[int] = Field(None, ge=0)
    attributes: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    review_summary: Optional[str] = None
    reviews: Optional[List[str]] = None
    best_for: Optional[str] = None
    weight_raw: Optional[str] = None
    language: Optional[Language] = None


class ProductOut(ProductIn):
    """Product as returned by GET /products/{id}."""

    currency: str = "INR"
    weight_g: Optional[float] = None
    in_stock: bool = False
    updated_at: Optional[str] = None


class IngestRequest(BaseModel):
    products: List[ProductPatch] = Field(..., min_length=1, max_length=1000)


class IngestResult(BaseModel):
    inserted: int = 0
    updated: int = 0
    product_ids: List[str] = Field(default_factory=list)
    index_updated: bool = False
    errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Shoppers
# ---------------------------------------------------------------------------
class Customer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    customer_id: str = Field(..., pattern=CUSTOMER_ID_PATTERN)
    name: str
    city: Optional[str] = None
    preferred_language: Language = "en"
    budget_min: Optional[float] = Field(None, ge=0)
    budget_max: Optional[float] = Field(None, ge=0)
    preferred_brands: List[str] = Field(default_factory=list)
    preferred_categories: List[str] = Field(default_factory=list)
    sizes: Dict[str, Any] = Field(default_factory=dict)   # {"shoe": 9, "apparel": "M"}
    persona_note: Optional[str] = None


class CustomerSummary(BaseModel):
    """GET /customers - only for the demo 'logged in as' selector. No orders exposed."""

    customer_id: str
    name: str
    preferred_language: Language = "en"


class Order(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_id: str = Field(..., pattern=ORDER_ID_PATTERN)
    customer_id: str = Field(..., pattern=CUSTOMER_ID_PATTERN)
    product_id: str = Field(..., pattern=PRODUCT_ID_PATTERN)
    quantity: int = Field(1, ge=1)
    price_paid: Optional[float] = Field(None, ge=0)
    order_date: str                                        # ISO date "2026-10-02"


# ---------------------------------------------------------------------------
# AI pipeline internals (Member 1 <-> Member 2)
# ---------------------------------------------------------------------------
class Constraints(BaseModel):
    """What the shopper wants. Filled by intent extraction, merged with session memory,
    and applied EXACTLY by Python tools (never by the LLM)."""

    category: Optional[str] = None
    subcategory: Optional[str] = None
    budget_min: Optional[float] = Field(None, ge=0)
    budget_max: Optional[float] = Field(None, ge=0)
    brands: List[str] = Field(default_factory=list)          # include only these
    exclude_brands: List[str] = Field(default_factory=list)
    size: Optional[str] = None                               # stored as text: "9", "M"
    color: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)        # English search words
    occasion: Optional[str] = None                           # "gift for father", "trekking"
    min_rating: Optional[float] = Field(None, ge=0, le=5)
    in_stock_only: bool = True
    sort_by: SortBy = "relevance"

    @field_validator("size", mode="before")
    @classmethod
    def _size_to_text(cls, v: Any) -> Optional[str]:
        if v is None or v == "":
            return None
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        return str(v).strip()


class Intent(BaseModel):
    """Structured understanding of one shopper message (output of LLM or mock parser)."""

    intent: IntentName = "search"
    constraints: Constraints = Field(default_factory=Constraints)
    search_query_en: str = ""                    # English rewrite used for retrieval
    refers_to_previous: bool = False             # "cheaper ones", "the first one"
    referenced_positions: List[int] = Field(default_factory=list)   # 1-based: "first two" -> [1, 2]
    product_ids_mentioned: List[str] = Field(default_factory=list)  # explicit "P-1004"
    attribute_asked: Optional[str] = None        # product_fact: "battery_hours", "warranty"
    language: Language = "en"
    confidence: float = Field(0.5, ge=0, le=1)


class StockResult(BaseModel):
    product_id: str
    found: bool
    in_stock: bool = False
    stock_qty: int = 0
    size_requested: Optional[str] = None
    size_available: Optional[bool] = None        # None = not asked / not applicable
    color_requested: Optional[str] = None
    color_match: Optional[bool] = None
    message: str = ""                            # plain-English fact written by code


class TurnRecord(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    answer_type: Optional[AnswerType] = None
    product_ids: List[str] = Field(default_factory=list)


class SessionState(BaseModel):
    """Conversation memory for one session, stored as JSON in sessions.state_json."""

    customer_id: str
    constraints: Constraints = Field(default_factory=Constraints)
    last_result_ids: List[str] = Field(default_factory=list)   # in display order
    clarify_count: int = 0                                     # consecutive clarifications
    asked_slots: List[str] = Field(default_factory=list)       # never re-ask these
    language: Language = "en"
    history: List[TurnRecord] = Field(default_factory=list)
    turn_count: int = 0


# ---------------------------------------------------------------------------
# Public API: POST /chat
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    customer_id: str = Field(..., pattern=CUSTOMER_ID_PATTERN)  # identity comes ONLY from here
    message: str = Field(..., min_length=1, max_length=1000)

    @field_validator("message")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be blank")
        return v


class ProductCard(BaseModel):
    product_id: str
    name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    in_stock: bool
    stock_qty: int = 0
    rating: Optional[float] = None
    reason: str = ""


class SourceRef(BaseModel):
    product_id: str
    fields: List[str] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    product_id: str
    name: str
    values: Dict[str, Any] = Field(default_factory=dict)


class ComparisonTable(BaseModel):
    fields: List[str] = Field(default_factory=list)
    rows: List[ComparisonRow] = Field(default_factory=list)


class ChatResponse(BaseModel):
    trace_id: str
    answer_type: AnswerType
    answer: str
    products: List[ProductCard] = Field(default_factory=list)
    sources: List[SourceRef] = Field(default_factory=list)
    follow_up_question: Optional[str] = None
    suggested_questions: List[str] = Field(default_factory=list)
    comparison: Optional[ComparisonTable] = None
    language: Language = "en"


# ---------------------------------------------------------------------------
# Audit / trace (R10)
# ---------------------------------------------------------------------------
class ToolCall(BaseModel):
    name: str                                    # "hybrid_retrieve", "apply_filters", ...
    input: Dict[str, Any] = Field(default_factory=dict)
    output_summary: str = ""
    duration_ms: float = 0.0


class TraceRecord(BaseModel):
    trace_id: str
    session_id: str
    customer_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    message: str
    language: Language = "en"
    intent: Dict[str, Any] = Field(default_factory=dict)
    retrieved: List[Dict[str, Any]] = Field(default_factory=list)   # [{product_id, score, source}]
    filters: Dict[str, Any] = Field(default_factory=dict)
    tools: List[ToolCall] = Field(default_factory=list)
    final_product_ids: List[str] = Field(default_factory=list)
    answer_type: AnswerType
    answer: str
    follow_up_question: Optional[str] = None
    response: Dict[str, Any] = Field(default_factory=dict)          # full ChatResponse
    latency_ms: float = 0.0
    llm_mode: str = "mock"


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    products: int = 0
    vector_index: int = 0
    mock_llm: bool = True
    llm_mode: str = "mock"
