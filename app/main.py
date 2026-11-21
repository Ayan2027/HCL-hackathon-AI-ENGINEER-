"""
ShopSense AI - FastAPI backend (Member 2).

Run:  uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs   (interactive Swagger UI for the judges)

Endpoints (official contract):
  POST /chat              shopper message -> grounded answer      (AI engine, Member 1)
  GET  /products/{id}     one product, straight from SQLite
  POST /ingest            add / update products, searchable at once
  GET  /trace/{trace_id}  replay how an answer was produced
  GET  /health            200 when ready
Extra helpers:
  GET  /products          browse / filter the catalogue (exact SQL-backed filters)
  GET  /customers         demo "logged in as" list (id + name only, no orders)
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.ai import orchestrator
from app.config import settings
from app.db import repository as repo
from app.db.bootstrap import load_if_empty
from app.db.catalog_import import catalogue_is_current
from app.db.database import init_db
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Constraints,
    CustomerSummary,
    HealthResponse,
    IngestRequest,
    IngestResult,
    ProductOut,
)
from app.tools.catalog_tools import apply_filters, sort_products

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("shopsense.api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    loaded = load_if_empty()
    logger.info("Data bootstrap: %s", loaded)
    if settings.catalog_json.exists() and not catalogue_is_current():
        logger.warning("The active catalogue does not match %s. Run: python -m app.db.catalog_import",
                       settings.catalog_json.name)
    try:
        orchestrator.startup()                       # build / load FAISS index
    except Exception:  # noqa: BLE001 - keyword search still works without vectors
        logger.exception("Vector index startup failed; continuing with keyword search")
    logger.info("ShopSense API ready (LLM mode: %s)", settings.llm_mode)
    yield


app = FastAPI(
    title="ShopSense AI",
    version=__version__,
    description="Conversational shopping assistant: grounded in the catalogue, safe, and traceable.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.0f}"
    return response


@app.exception_handler(Exception)
async def unhandled_error(_: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {type(exc).__name__}"})


def _product_out(product: dict) -> ProductOut:
    return ProductOut(**{k: v for k, v in product.items() if not k.startswith("_")})


# ---------------------------------------------------------------------------
# Official endpoints
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest) -> ChatResponse:
    """customer_id in the body is the ONLY source of identity (R9)."""
    return orchestrator.handle_chat(request)


@app.get("/products/{product_id}", response_model=ProductOut, tags=["catalogue"])
def get_product(product_id: str) -> ProductOut:
    product = repo.get_product(product_id.upper())
    if not product:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    return _product_out(product)


@app.post("/ingest", response_model=IngestResult, tags=["catalogue"])
def ingest(request: IngestRequest) -> IngestResult:
    """Add or update products. Keyword index updates in the same DB transaction;
    the vector index is updated right after, so the next /chat sees the change (R11)."""
    result = repo.upsert_products(request.products)
    if result.product_ids:
        try:
            result.index_updated = orchestrator.on_products_changed(result.product_ids)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"vector index update failed: {exc}")
    if not result.product_ids and result.errors:
        raise HTTPException(status_code=422, detail=result.errors)
    return result


@app.get("/trace/{trace_id}", tags=["audit"])
def get_trace(trace_id: str) -> dict:
    trace = repo.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return trace


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    products = repo.count_products()
    vectors = orchestrator.vector_index_size()
    return HealthResponse(
        status="ok" if products > 0 else "degraded",
        products=products,
        vector_index=vectors,
        mock_llm=settings.use_mock_llm,
        llm_mode=settings.llm_mode,
        retrieval_mode="hybrid" if vectors > 0 else "keyword",
        vector_status=None if vectors > 0 else orchestrator.vector_status(),
        archived_products=repo.count_archived_products(),
        inventory_simulated=repo.any_simulated_stock(),
    )


# ---------------------------------------------------------------------------
# Helpers for the UI and judges
# ---------------------------------------------------------------------------
@app.get("/products", response_model=List[ProductOut], tags=["catalogue"])
def list_products(
    category: Optional[str] = None,
    brand: Optional[str] = None,
    max_price: Optional[float] = Query(None, ge=0),
    min_price: Optional[float] = Query(None, ge=0),
    in_stock: bool = False,
    sort_by: str = Query("relevance", pattern="^(relevance|price_asc|price_desc|rating)$"),
    limit: int = Query(50, ge=1, le=500),
) -> List[ProductOut]:
    """Browse the catalogue with the SAME exact filters the assistant uses."""
    constraints = Constraints(category=category, brands=[brand] if brand else [], budget_max=max_price,
                              budget_min=min_price, in_stock_only=in_stock, sort_by=sort_by)
    kept, _ = apply_filters(repo.get_all_products(), constraints)
    return [_product_out(p) for p in sort_products(kept, constraints.sort_by)[:limit]]


@app.get("/customers", response_model=List[CustomerSummary], tags=["demo"])
def list_customers() -> List[CustomerSummary]:
    """Simulates a login picker for the demo. Exposes no orders or personal details."""
    return [CustomerSummary(**c) for c in repo.list_customers()]
