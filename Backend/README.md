# Backend — AdaptRoute Inference Pipeline

FastAPI/Flask server exposing the AdaptRoute pipeline for local and cloud deployment.

## Quick Start

```bash
# Install deps
pip install flask flask-cors pyngrok transformers peft bitsandbytes accelerate

# Run pipeline
python pipeline.py
```

## Deployment

### Colab (Recommended)

Use `worker_colab.py` to expose `pipeline.py` via ngrok:

1. Run your `pipeline.py` cell first (loads models)
2. Append `worker_colab.py` code to the same cell
3. It will print a public ngrok URL for remote access

```bash
POST https://<ngrok-url>/generate
Body: { "query": "Write a Python function to sort a list" }
```

### Local Server

```bash
python pipeline.py
# Serves at http://localhost:5000
```

## Hardware Requirements

### Inference

| Hardware           | VRAM        | Performance      | Best For                    |
| ------------------ | ----------- | ---------------- | --------------------------- |
| **NVIDIA T4**      | 16 GB       | ~2-3 sec/query   | Colab, Cloud inference      |
| **RTX 3070/4070**  | 8-12 GB     | ~1-2 sec/query   | Local/personal machine      |
| **RTX 4090**       | 24 GB       | ~0.5 sec/query   | Fast local inference        |
| **Apple M1/M2/M3** | Unified RAM | ~3-5 sec/query   | MacOS with NPU acceleration |
| **CPU only**       | -           | ~20-60 sec/query | Batch processing, no GPU    |

### Setup

- Base model (Qwen2.5-1.5B @ 4-bit): 4 GB
- Firewall (DeBERTa-v3): 1.5 GB
- Gating network (DistilBERT): 250 MB
- All adapters (4 domains): 200 MB
- **Total:** ~6 GB VRAM recommended minimum

## API Endpoints

### `GET /health`

Health check.

```bash
curl http://localhost:5000/health
```

Response:

```json
{ "status": "ok", "message": "Pipeline running." }
```

### `POST /generate`

Generate response with router. **Returns BOTH base model response and adapted response side-by-side.**

**Request body:**

```json
{
  "query": "What is the Python syntax for list comprehension?"
}
```

**Response:**

```json
{
  "status": "success",
  "base_response": "List comprehension is a way to create lists in Python...",
  "adapted_response": "In Python, list comprehensions provide a concise syntax for creating lists. For example: [x*2 for x in range(10)]",
  "adapter_used": "code",
  "gating_scores": {
    "code": 0.85,
    "math": 0.1,
    "qa": 0.03,
    "medical": 0.02
  },
  "firewall_label": "SAFE",
  "time_taken_seconds": 2.34
}
```

#### Response Fields

| Field                | Description                                                         |
| -------------------- | ------------------------------------------------------------------- |
| `status`             | `success` or `blocked` or `error`                                   |
| `base_response`      | Answer from 1.5B model alone (no adapters) — generic/mediocre       |
| `adapted_response`   | Answer from 1.5B model + selected LoRA adapter — domain-specialized |
| `adapter_used`       | Which domain was routed to: `code`, `math`, `qa`, or `medical`      |
| `gating_scores`      | Softmax probabilities for each domain (routing confidence)          |
| `firewall_label`     | Prompt injection detection result: `SAFE` or `INJECTION`            |
| `time_taken_seconds` | Total latency (firewall + gating + generation)                      |

#### Hackathon Key Insight

The side-by-side comparison (**`base_response` vs `adapted_response`**) visually demonstrates:

- ✓ Base model is **generic** (decent but not specialized)
- ✓ Adapted model is **expert** (specialized via LoRA)
- ✓ Why adapters matter for edge: **Multiple specialists, single storage footprint**

## Files

- `pipeline.py` — Core inference logic (firewall → gating → adapter merge → generation)
- `worker_colab.py` — Flask wrapper for Colab deployment with ngrok tunneling
