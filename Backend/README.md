# Backend — AdaptRoute Inference Pipeline

FastAPI server exposing the AdaptRoute pipeline for SSH / H200 deployment.

## Quick Start (on SSH server)

```bash
cd Backend
pip install -r requirements.txt
python app.py
```

Server starts at `http://0.0.0.0:8000`. Override with `PORT=9000 python app.py`.

On first run, adapter weights are automatically downloaded from HuggingFace.

## Architecture

```
app.py          ← FastAPI wrapper (endpoints + CORS + startup)
pipeline.py     ← Core inference logic (firewall → gating → adapter → generate)
```

`app.py` imports `pipeline.py` directly — no modifications to the pipeline needed.

## API Endpoints

### `GET /health`

Health check.

```bash
curl http://<server-ip>:8000/health
```

Response:

```json
{ "status": "ok", "message": "Pipeline running." }
```

### `POST /generate`

Generate response with routing.

```bash
curl -X POST http://<server-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "Write a Python function to sort a list"}'
```

Response:

```json
{
  "status": "success",
  "response": "def sort_list(lst): ...",
  "adapter_used": "code",
  "gating_scores": {
    "code": 0.85,
    "math": 0.10,
    "qa": 0.03,
    "medical": 0.02
  },
  "firewall_label": "SAFE",
  "time_seconds": 1.42
}
```

### Interactive Docs

FastAPI auto-generates docs at:
- Swagger UI: `http://<server-ip>:8000/docs`
- ReDoc: `http://<server-ip>:8000/redoc`

## Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server — endpoints, CORS, model loading on startup |
| `pipeline.py` | Core inference (firewall → gating → adapter merge → generation) |
| `pipeline_vllm.py` | Alternative pipeline using vLLM (optional, faster) |
| `train_firewall_v3.py` | Firewall test script |
| `requirements.txt` | Python dependencies |
