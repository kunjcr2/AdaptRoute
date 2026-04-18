"""
AdaptRoute FastAPI Server
~~~~~~~~~~~~~~~~~~~~~~~~~
Thin wrapper around pipeline.py.

Endpoints:
    POST /generate         - non-streaming, returns full response as JSON
    POST /generate/stream  - streaming, returns NDJSON (one JSON object per line)
    POST /train            - trigger retraining loop
    GET  /train/status     - training status
    POST /train/stop       - halt training
    GET  /health           - health check
    GET  /stats            - query log count

Every successful query (streamed or not) is appended to query_log.jsonl
for the continual learning loop (online_train.py).
"""

import os
import json
import time
import threading
from contextlib import asynccontextmanager
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

import pipeline
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
    print("  Server ready for requests")
    print("=" * 60)
    yield
    print("AdaptRoute — Shutting down")


app = FastAPI(
    title="AdaptRoute",
    description="Task-aware routing & LoRA-adapted inference pipeline.",
    version="2.1.0",
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


@app.get("/stats", response_model=StatsResponse)
async def stats():
    count = 0
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            count = sum(1 for line in f if line.strip())
    return StatsResponse(query_count=count)


@app.post("/generate", response_model=QueryResponse)
async def generate(req: QueryRequest):
    """Non-streaming. Returns the full response as JSON once generation finishes."""
    result = pipeline.process_query(req.query)

    if result.get("status") == "success" and result.get("response"):
        try:
            log_query(
                question=req.query,
                model_response=result["response"],
                domain=result.get("adapter_used", "general"),
            )
        except Exception as e:
            print(f"[Log] Warning: failed to write query log — {e}")

    return QueryResponse(**result)


@app.post("/generate/stream")
async def generate_stream(req: QueryRequest):
    """
    Streaming. Returns NDJSON: one JSON object per line, terminated with \\n.

    Frontend usage (fetch + ReadableStream):
        const res = await fetch('/generate/stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: '...'})
        });
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            buf += decoder.decode(value, {stream: true});
            const lines = buf.split('\\n');
            buf = lines.pop();  // keep incomplete last line
            for (const line of lines) {
                if (!line) continue;
                const obj = JSON.parse(line);
                // obj.type is one of: meta, token, done, blocked, error
                if (obj.type === 'token') appendToUI(obj.text);
                else if (obj.type === 'meta') showRoutingBadge(obj);
                else if (obj.type === 'done') markComplete(obj);
                else if (obj.type === 'blocked') showBlocked(obj.message);
            }
        }
    """
    query = req.query

    def event_stream():
        full_response = ""
        routing_meta  = {}

        try:
            for chunk in pipeline.process_query_stream(query):
                # Track metadata + full response so we can log after the stream ends
                if chunk.get("type") == "meta":
                    routing_meta = chunk
                elif chunk.get("type") == "done":
                    full_response = chunk.get("full_response", "")

                yield json.dumps(chunk) + "\n"

            # Log after streaming completes successfully
            if full_response and routing_meta:
                try:
                    log_query(
                        question=query,
                        model_response=full_response,
                        domain=routing_meta.get("adapter_used", "general"),
                    )
                except Exception as e:
                    print(f"[Log] Warning: failed to write query log — {e}")

        except Exception as e:
            # Final-chance error channel
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind a proxy
        },
    )


@app.post("/train", response_model=TrainResponse)
async def train():
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
    with _training_lock:
        s = dict(_training_status)
    return TrainStatusResponse(**s)


@app.post("/train/stop", response_model=TrainResponse)
async def train_stop():
    with _training_lock:
        active = _training_active
    if not active:
        return TrainResponse(status="idle", message="No training in progress.")
    _stop_event.set()
    return TrainResponse(status="stopping", message="Stop signal sent — will halt after current step.")


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7180"))
    uvicorn.run("app:app", host=host, port=port, log_level="info")