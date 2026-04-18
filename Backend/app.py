"""
AdaptRoute FastAPI Server
~~~~~~~~~~~~~~~~~~~~~~~~~
Thin wrapper around pipeline.py — exposes the existing
prepare() / load_all_models() / process_query() functions as REST endpoints.

Every successful query is appended to query_log.jsonl for the
continual learning loop (online_train.py).

Usage (on the SSH / H200 server):
    cd Backend
    python app.py                 # starts on 0.0.0.0:7180
    PORT=9000 python app.py       # override port

Compatible with Python 3.11.
"""

import os
import time
import threading
from contextlib import asynccontextmanager
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ── Import existing pipeline (unchanged) ────────────────────────
import pipeline

# ── Import logger, retraining loop, and log path ────────────────
from online_train import log_query, run_retraining_loop, LOG_FILE


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


class TrainResponse(BaseModel):
    status: str
    message: str


class TrainStatusResponse(BaseModel):
    active: bool
    phase: str
    current_domain: Optional[str] = None
    domains_done: List[str] = []
    started_at: Optional[float] = None


class StatsResponse(BaseModel):
    query_count: int


# ── Training state ──────────────────────────────────────────────

_training_lock = threading.Lock()
_training_active = False
_stop_event = threading.Event()
_training_status: dict = {
    "active": False,
    "phase": "idle",
    "current_domain": None,
    "domains_done": [],
    "started_at": None,
}


def _on_phase(phase: str, domain: Optional[str] = None):
    global _training_status
    with _training_lock:
        if phase == "loading":
            _training_status["phase"] = "loading"
        elif phase == "scoring":
            _training_status["phase"] = "scoring"
        elif phase == "training":
            _training_status["phase"] = "training"
            _training_status["current_domain"] = domain
        elif phase in ("done_domain", "skipped"):
            if domain:
                _training_status["domains_done"].append(domain)
            _training_status["current_domain"] = None
        elif phase in ("complete", "stopped"):
            _training_status["phase"] = phase
            _training_status["current_domain"] = None


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


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Return the number of queries logged in query_log.jsonl."""
    count = 0
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            count = sum(1 for line in f if line.strip())
    return StatsResponse(query_count=count)


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


@app.post("/train", response_model=TrainResponse)
async def train():
    """
    Trigger a full GRPO retraining cycle for all domain adapters.
    Runs in a background thread and returns immediately.
    Returns 409 if training is already in progress.
    """
    global _training_active, _training_status

    with _training_lock:
        if _training_active:
            raise HTTPException(status_code=409, detail="Training already in progress.")
        _training_active = True
        _stop_event.clear()
        _training_status = {
            "active": True,
            "phase": "loading",
            "current_domain": None,
            "domains_done": [],
            "started_at": time.time(),
        }

    def _run():
        global _training_active, _training_status
        try:
            run_retraining_loop(stop_event=_stop_event, on_phase=_on_phase)
        finally:
            with _training_lock:
                _training_active = False
                _training_status["active"] = False

    threading.Thread(target=_run, daemon=True).start()
    return TrainResponse(status="started", message="Retraining loop launched in background.")


@app.get("/train/status", response_model=TrainStatusResponse)
async def train_status():
    """Return the current training state (phase, domain, progress)."""
    with _training_lock:
        s = dict(_training_status)
    return TrainStatusResponse(**s)


@app.post("/train/stop", response_model=TrainResponse)
async def train_stop():
    """Send a stop signal to the running training loop."""
    with _training_lock:
        active = _training_active
    if not active:
        return TrainResponse(status="idle", message="No training in progress.")
    _stop_event.set()
    return TrainResponse(status="stopping", message="Stop signal sent — will halt after current step.")


# ── Run ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7180"))
    uvicorn.run("app:app", host=host, port=port, log_level="info")
