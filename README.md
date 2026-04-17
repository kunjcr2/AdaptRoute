# AdaptRoute — Task-Aware SLM Router with Dynamic LoRA Switching

> A learned gating network that dynamically routes queries to domain-specific LoRA expert adapters at inference time — MoE-style routing without end-to-end training cost. One frozen base model, multiple lightweight experts, zero cloud dependency.

---

## What Is This?

Small language models (SLMs) are the practical option for real-world deployment — local inference, edge devices, anything without a data center. But SLMs are generalists that do okay at everything and excel at nothing. A 1.5B parameter model asked to write production Python, solve a differential equation, and answer a medical question will struggle at all three.

AdaptRoute fixes this: **one frozen base model** + **multiple lightweight LoRA adapters** (a few MB each) + **a fast gating network** that decides which adapter to use per query. The system gets the equivalent of multiple specialized models for the cost of one.

```
User query
    │
    ▼
┌──────────────────────────────────────┐
│   🛡️  Firewall (DeBERTa-v3)         │
│   Binary: SAFE | INJECTION           │  ← blocks prompt injections
└──────────────────────────────────────┘
    │  (if safe)
    ▼
┌──────────────────────────────────────┐
│   🧠  Gating Network (DistilBERT)   │
│   5-class: code|math|qa|medical|     │  ← task routing
│            general                    │
└──────────────────────────────────────┘
    │
    ├── code/math/qa/medical → Load domain LoRA adapter
    ├── general              → Use base model (no adapter)
    │
    ▼
┌──────────────────────────────────────┐
│   Qwen2.5-1.5B (bfloat16)           │
│   + dynamically loaded LoRA adapter  │  ← generation
└──────────────────────────────────────┘
    │
    ▼
  Response
```

---

## Project Structure

```
AdaptRoute/
├── Backend/
│   ├── app.py                 # FastAPI server (wraps pipeline)
│   ├── pipeline.py            # HF Transformers inference pipeline
│   ├── pipeline_vllm.py       # vLLM inference pipeline (10x faster)
│   └── requirements.txt
├── Frontend/
│   └── src/
│       ├── pages/
│       │   ├── Home.jsx       # Landing page
│       │   ├── Demo.jsx       # Live inference demo
│       │   ├── Architecture.jsx
│       │   ├── Evaluation.jsx
│       │   └── Firewall.jsx
│       └── lib/
│           └── adaptroute.js  # API client
├── notebooks/
│   ├── train_gating_v2.py     # Gating network training (5-class, H200)
│   ├── Adapters_Training_v5.py # LoRA adapter SFT pipeline
│   ├── train_firewall_v3.py   # Firewall training
│   └── Eval.ipynb             # Evaluation notebook
├── datasets/
│   ├── code.json              # 7k code queries
│   ├── math.json              # 7.5k math queries
│   ├── qa.json                # 10k QA queries
│   └── medical.json           # 10k medical queries
└── README.md
```

---

## Architecture

### Firewall — Prompt Injection Detection

| Component | Details |
|-----------|---------|
| Model | [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2 ) |
| Type | Binary classifier (SAFE / INJECTION) |
| Size | ~180 MB |
| Runs on | CPU (keeps GPU free for generation) |

Blocks adversarial prompt injections, jailbreaks, and hidden system-note attacks before they reach the gating network.

### Gating Network — Task Router

| Component | Details |
|-----------|---------|
| Model | [kunjcr2/gating-bert-adaptroute](https://huggingface.co/kunjcr2/gating-bert-adaptroute ) |
| Base | `distilbert-base-uncased` + LoRA (merged) |
| Classes | `code` · `math` · `qa` · `medical` · `general` |
| Latency | ~5ms on CPU |
| Training | `notebooks/train_gating_v2.py` |

The 5th `general` class ensures queries like *"How do I open a tight jar?"* or *"What is the history of the internet?"* get routed to the base model without any adapter — instead of being misclassified into a specialist domain.

### LoRA Expert Adapters

Four domain-specific adapters trained via SFT on Qwen2.5-1.5B:

| Adapter | HuggingFace Repo | Domain |
|---------|-------------------|--------|
| `lora-code` | [kunjcr2/code-adaptroute-v3](https://huggingface.co/kunjcr2/code-adaptroute-v3 ) | Python code generation |
| `lora-math` | [kunjcr2/math-adaptroute-v3](https://huggingface.co/kunjcr2/math-adaptroute-v3 ) | Step-by-step math reasoning |
| `lora-qa` | [kunjcr2/qa-adaptroute-v3](https://huggingface.co/kunjcr2/qa-adaptroute-v3 ) | Factual question answering |
| `lora-medical` | [kunjcr2/medical-adaptroute-v3](https://huggingface.co/kunjcr2/medical-adaptroute-v3 ) | Clinical triage & medical Q&A |

### Base Model

| Model | Params | Precision | Attention |
|-------|--------|-----------|-----------|
| [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B ) | 1.5B | bfloat16 | SDPA (Flash Attention) |

Weights are fully frozen. Only LoRA adapters are trained.

---

## Quick Start

### Backend (SSH / GPU Server)

```bash
# 1. Create working directory
mkdir Backend && cd Backend

# 2. Install dependencies
pip install fastapi uvicorn[standard] torch transformers peft accelerate huggingface_hub

# 3. Create pipeline.py and app.py (copy from repo)
nano pipeline.py
nano app.py

# 4. Set HuggingFace token and run
export HF_TOKEN="hf_your_token_here"
python app.py
```

The server starts on `0.0.0.0:7180`, downloads all adapter weights on first run.

### Frontend (Local)

```bash
cd Frontend
npm install
npm run dev
```

Update `WORKER_URL` in `Frontend/src/lib/adaptroute.js` to point to your server IP.

### Test

```bash
# Health check
curl http://your-server:7180/health

# Inference
curl -X POST http://your-server:7180/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "Write a Python function to sort a list"}'
```

---

## Backend Options

### Standard Pipeline (`pipeline.py`)
Uses HuggingFace Transformers for generation. Simpler setup, compatible with any GPU.

### vLLM Pipeline (`pipeline_vllm.py`)
Uses [vLLM](https://github.com/vllm-project/vllm ) for ~10x faster inference with native LoRA hot-swapping. Recommended for production.

To switch: change `import pipeline` → `import pipeline_vllm as pipeline` in `app.py`.

---

## Training

### Gating Network (5-class)

```bash
cd notebooks
export HF_TOKEN="hf_..."
python train_gating_v2.py
```

Trains a DistilBERT + LoRA classifier with 5 classes. The `general` class is built from `tatsu-lab/alpaca` + handcrafted queries to prevent misrouting.

**H200 optimised:** batch_size=128, bf16, 4 dataloader workers.

### LoRA Adapters

```bash
cd notebooks
python Adapters_Training_v5.py
```

Trains all 4 domain adapters sequentially via SFT. Uses LoRA with:

| Parameter | Value |
|-----------|-------|
| Rank (`r`) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj` |
| Precision | bfloat16 |

---

## API

### `GET /health`
Returns server status and loaded model info.

### `POST /generate`
```json
// Request
{ "query": "What are the symptoms of diabetes?" }

// Response
{
  "status": "success",
  "response": "The common symptoms of diabetes include...",
  "adapter_used": "medical",
  "gating_confidence": 0.9987,
  "gating_scores": {
    "code": 0.0001,
    "math": 0.0002,
    "qa": 0.0010,
    "medical": 0.9987,
    "general": 0.0000
  },
  "firewall_label": "SAFE",
  "time_seconds": 2.14
}
```

---

## Connection to MoE

AdaptRoute is a decoupled Mixture of Experts — the routing gate and experts are trained independently, making it practical without multi-GPU infrastructure.

|                  | MoE (standard)           | AdaptRoute                   |
| ---------------- | ------------------------ | ---------------------------- |
| Gate training    | Joint with experts       | Independent classifier       |
| Expert training  | Joint, end-to-end        | Independent SFT per adapter  |
| Routing          | Soft (differentiable)    | Hard routing by gate output  |
| Hardware         | Multi-GPU required       | Single GPU sufficient        |
| Expert swap cost | Cannot swap at inference | Free — adapters are files    |

---

## Tech Stack

| Layer | Library | Role |
|-------|---------|------|
| Inference server | `FastAPI` + `uvicorn` | REST API for frontend |
| Generation | `transformers` / `vllm` | Base model + adapter inference |
| Adapter management | `peft` | LoRA loading, merging, swapping |
| Training | `trl` — `SFTTrainer` | Supervised fine-tuning |
| Frontend | React + Vite | Demo UI with live inference |
| Compute | NVIDIA H200 (SSH) | Training & inference |

---

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685 )
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088 )
- [PEFT library](https://huggingface.co/docs/peft )
- [vLLM](https://github.com/vllm-project/vllm )
- [Qwen2.5 model family](https://huggingface.co/Qwen )
- [ProtectAI prompt injection detector](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2 )