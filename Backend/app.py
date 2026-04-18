"""
AdaptRoute FastAPI Server
~~~~~~~~~~~~~~~~~~~~~~~~~
Thin wrapper around pipeline.py.

Endpoints:
    POST /generate         - non-streaming, returns full response as JSON
    POST /generate/stream  - streaming, returns NDJSON (one JSON object per line)
    GET  /health           - health check
"""

import os
from contextlib import asynccontextmanager
from typing import Optional, Dict
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

import pipeline


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
async def lifespan(_: FastAPI):
    print("=" * 60)
    print("  AdaptRoute — Downloading adapters (if needed) ...")
    print("=" * 60)
    pipeline.prepare()

    print("=" * 60)
    print("  AdaptRoute — Loading all models ...")
    print("=" * 60)
    pipeline.load_all_models()

    print("=" * 60)
    print("  Server ready for requests")
    print("=" * 60)
    yield
    print("AdaptRoute — Shutting down")


app = FastAPI(
    title="AdaptRoute",
    description="Task-aware routing & LoRA-adapted inference pipeline.",
    version="2.2.0",
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
    return HealthResponse(status="ok", message="Pipeline running.")


@app.post("/generate", response_model=QueryResponse)
async def generate(req: QueryRequest):
    """Non-streaming. Returns the full response as JSON once generation finishes."""
    result = pipeline.process_query(req.query)
    return QueryResponse(**result)


@app.post("/generate/stream")
async def generate_stream(req: QueryRequest):
    """
    Streaming. Returns NDJSON: one JSON object per line, terminated with \\n.

    Line types:
      {"type": "meta", "adapter_used": ..., "routing_mode": ..., ...}
      {"type": "token", "text": "..."}
      {"type": "done", "time_seconds": ..., "full_response": "..."}
      {"type": "blocked", "message": "...", "firewall_label": "INJECTION"}
      {"type": "error", "message": "..."}
    """
    query = req.query

    def event_stream():
        try:
            for chunk in pipeline.process_query_stream(query):
                yield json.dumps(chunk) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind a proxy
        },
    )


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7180"))
    uvicorn.run("app:app", host=host, port=port, log_level="info")