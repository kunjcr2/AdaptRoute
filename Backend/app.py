"""
AdaptRoute FastAPI Server
~~~~~~~~~~~~~~~~~~~~~~~~~
Thin wrapper around pipeline.py — exposes the existing
prepare() / load_all_models() / process_query() functions as REST endpoints.

Every successful query is appended to query_log.jsonl for the
continual learning loop (continual_learning.py).

Usage (on the SSH / H200 server):
    cd Backend
    python app.py                 # starts on 0.0.0.0:7180
    PORT=9000 python app.py       # override port

Compatible with Python 3.11.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ── Import existing pipeline (unchanged) ────────────────────────
import pipeline

# ── Import logger from continual learning loop ──────────────────
from online_train import log_query


# ── Pydantic schemas ────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query to process")


class QueryResponse(BaseModel):
    status: str
    response: Optional[str] = None
    message: Optional[str] = None
    adapter_used: Optional[str] = None
    routing_mode: Optional[str] = None
    gating_confidence: Optional[float] = None
    gating_scores: Optional[Dict[str, float]] = None
    firewall_label: Optional[str] = None
    time_seconds: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    message: str


# ── Lifespan: load models on startup ───────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  AdaptRoute — Downloading adapters (if needed) ...")
    print("=" * 60)
    pipeline.prepare()

    print("=" * 60)
    print("  AdaptRoute — Loading all models ...")
    print("=" * 60)
    pipeline.load_all_models()

    print("=" * 60)
    print("  ✓ Server ready for requests")
    print("=" * 60)
    yield
    print("AdaptRoute — Shutting down")


# ── FastAPI app ─────────────────────────────────────────────────

app = FastAPI(
    title="AdaptRoute",
    description=(
        "Task-aware routing & LoRA-adapted inference pipeline. "
        "Firewall → Gating Network → Dynamic Adapter → Generation."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return HealthResponse(status="ok", message="Pipeline running.")


@app.post("/generate", response_model=QueryResponse)
async def generate(req: QueryRequest):
    """
    Run a query through the full AdaptRoute pipeline:
    Firewall → Gating → Adapter → Generation.

    Every successful (non-blocked) response is appended to
    query_log.jsonl for the continual learning loop.
    """
    result = pipeline.process_query(req.query)

    # Log every successful response for continual learning.
    # Blocked / error responses are not logged — no useful signal.
    if result.get("status") == "success" and result.get("response"):
        try:
            log_query(
                question=req.query,
                model_response=result["response"],
                domain=result.get("adapter_used", "general"),
            )
        except Exception as e:
            # Never let logging crash the API response
            print(f"[Log] Warning: failed to write query log — {e}")

    return QueryResponse(**result)


# ── Run ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7180"))
    uvicorn.run("app:app", host=host, port=port, log_level="info")