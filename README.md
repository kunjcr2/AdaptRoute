# AdaptRoute — Task-Aware SLM Router with Soft LoRA Merging

> A learned gating network that dynamically blends LoRA expert adapters at inference time — sparse MoE-style routing without the end-to-end training cost. Built for edge devices where small models must punch above their weight.

---

## What Is This?

Small language models (SLMs) are the only practical option for edge deployment — phones, embedded systems, local inference boxes, anything without a data center behind it. But SLMs have a well-known problem: they are generalists that do okay at everything and excel at nothing. A 1.5B parameter model asked to write production Python, solve a differential equation, and summarize a legal document will struggle at all three.

The standard answer is task-specific fine-tuning — but that means shipping a separate model per task, which blows the memory and storage budget on any real edge device.

AdaptRoute solves this differently: one frozen base model lives on the device, several lightweight LoRA adapters (each a few MB) are stored alongside it, and a fast gating network running entirely on-device decides which adapters to blend for each query. The device gets the equivalent of multiple specialized models for the cost of one — with a routing layer small enough to run on CPU in under 10ms.

This is conceptually identical to the routing gate in a Mixture-of-Experts (MoE) model — except decoupled from training, making it practical to build, deploy, and run without multi-GPU infrastructure.

```
User query
    │
    ▼
┌─────────────────────────────────────────────┐
│          Firewall (DistilBERT)              │
│                                             │
│  Binary: [Clear] | [Malicious/OOD]          │  <- input filtering
└─────────────────────────────────────────────┘
    │
    │ (If Cleared)
    ▼
┌─────────────────────────────────────────────┐
│          Gating network (DistilBERT)        │
│                                             │
│  p(code), p(math), p(qa), p(medical)        │  <- task routing
└─────────────────────────────────────────────┘
    │
    ▼
Soft adapter merge  <-  add_weighted_adapter(adapters, weights)
    │
    ▼
Base SLM + merged LoRA adapter
    │
    ▼
Response
```

**Hard routing** (pick one adapter) is the argmax version. **Soft routing** (blend top-k adapters by probability) is what this project implements — it handles ambiguous queries gracefully and maps directly to the MoE gating mechanism.

> **Edge device motivation:** LoRA adapters for a 1.5B model are typically 10–40 MB each. Storing four domain adapters costs less than 200 MB total — a fraction of the base model's footprint. The gating network (DistilBERT) is 250 MB and runs on CPU. The entire system fits on a mid-range mobile device or single-board computer, with no cloud dependency.

---

## Project Stages

| Stage | What Happens | Estimated Time (A100) |
|---|---|---|
| 1. Firewall | Fine-tune DistilBERT for adversarial/OOD detection | ~10 min |
| 2. Gating network | Fine-tune DistilBERT as a 4-class task classifier | ~10 min |
| 3. LoRA adapters | SFT one adapter per domain on top of the base SLM | ~25 min per adapter |
| 4. Soft merge | Wire gate probabilities into `add_weighted_adapter()` | inference-time only |
| 5. Eval + demo | Benchmark quality delta, build Gradio UI | ~1–2 hours |

---

## Architecture

### Base Model

| Model | Params | VRAM (4-bit) | Notes |
|---|---|---|---|
| `Qwen2.5-1.5B` | 1.5B | ~4 GB | Primary choice — fast SFT, strong instruction following |

Loaded in 4-bit NF4 quantization via `bitsandbytes`. Weights are fully frozen. Only the LoRA adapters are trained.

---

### Firewall

A binary classifier (also `distilbert-base-uncased`) that acts as the first line of defense. It identifies queries that are either adversarial (jailbreaks, prompt injections) or completely out-of-domain (OOD).

**Model Path:** [kunjcr2/firewall-bert-adaptroute](https://huggingface.co/kunjcr2/firewall-bert-adaptroute)

**Why a separate model:** By decoupling the firewall from the gating network, we can update security policies or adversarial training sets without retraining the task-specific routing logic. It ensures that the gating network only processes "clean" queries, improving overall system stability.

---

### Gating Network

A multiclass classifier built on `distilbert-base-uncased`. Takes the raw user query as input and outputs a softmax distribution over **4 domain-specific classes**.

**Model Path:** [kunjcr2/gating-bert-adaptroute](https://huggingface.co/kunjcr2/gating-bert-adaptroute)

**Classes:** `code` · `math` · `qa` · `medical`

**Why DistilBERT:** 66M parameters, ~5ms inference on CPU. On an edge device, the gate runs before every query—it must add near-zero latency or it becomes the bottleneck.

By focusing on high-signal specialized domains, the routing network inherently acts as a robust filter. Queries that don't align with the primary expert domains are assigned low probability across the board. This specialization reduces the attack surface compared to a general-purpose SLM.

---

### LoRA Expert Adapters

Four domain-specific adapters trained via SFT on the base model. Each adapter is a small set of low-rank weight matrices injected into the attention layers.

| Adapter | Target domain |
|---|---|
| `lora-code` | Python code generation and explanation |
| `lora-math` | Step-by-step mathematical reasoning |
| `lora-qa` | Grounded question answering from context |
| `lora-medical` | Clinical triage and medical Q&A |

At inference, the gate probabilities are used as weights to blend these adapters via `peft`'s `add_weighted_adapter()`. The merged adapter is loaded onto the frozen base model for the forward pass.

---

## Datasets

### Gating Network Training

| Dataset | Label | Source |
|---|---|---|
| `code.json` (~10k sample) | `code` | `codeparrot/github-code` |
| `math.json` (~10k sample) | `math` | `lighteval/MATH` |
| `qa.json` (~10k sample) | `qa` | `rajpurkar/squad` |
| `medical.json` (~10k sample) | `medical` | `lavita/ChatDoctor-HealthCareMagic-100k` |

### LoRA Adapter SFT

| Adapter | Dataset | Size |
|---|---|---|
| `lora-code` | `code.json` | ~10k instruction-response pairs |
| `lora-math` | `math.json` | ~10k problems with step-by-step solutions |
| `lora-qa` | `qa.json` | ~10k Q&A pairs |
| `lora-medical` | `medical.json` | ~10k clinical Q&A pairs |

All datasets are loaded locally as JSON files from the `datasets/` directory.

---

## Tech Stack

| Layer | Library | Role |
|---|---|---|
| Fine-tuning | `transformers` + `peft` | `LoraConfig`, `get_peft_model`, SFT for both gate and adapters |
| Adapter merge | `peft` | `add_weighted_adapter()` blends adapters by gate probabilities |
| Training loop | `trl` — `SFTTrainer` | Dataset formatting, packing, trainer config |
| Quantization | `bitsandbytes` | 4-bit NF4 base model loading, keeps VRAM under 6 GB |
| Data loading | `datasets` | HuggingFace datasets hub |
| Experiment tracking | `wandb` | Loss curves, eval metrics, adapter comparison |
| Demo UI | `gradio` | Query in → gate label + merged response out |
| Compute | Colab A100 (40 GB) | ~25 min per adapter SFT at 1.5B scale |

---

## LoRA Configuration

Same config applied to all four domain adapters.

| Parameter | Value | Reason |
|---|---|---|
| rank (`r`) | `8` | Compact, fast to merge; sufficient capacity for domain shift at 1.5B |
| alpha | `16` | 2× rank — standard stable scaling |
| dropout | `0.05` | Light regularization for small dataset sizes |
| target modules | `q, k, v, o` | Attention projections only — faster training |
| precision | `BF16` | A100 native, stable for SFT |

---

## How Soft Routing Works

This is the key mechanism that makes AdaptRoute different from a simple classifier-to-adapter pipeline.

```python
from peft import PeftModel

# 1. Firewall check (binary model)
is_clean = firewall_model(query)
if not is_clean:
    return "Query blocked by firewall."

# 2. Gate outputs softmax probabilities
probs = gate_model(query)
# e.g. {"code": 0.72, "math": 0.21, "qa": 0.05, "medical": 0.02}

# Take top-2 adapters and their weights
top_adapters = ["lora-code", "lora-math"]
top_weights   = [0.72, 0.21]

# Merge adapters weighted by gate output
model.add_weighted_adapter(
    adapters=top_adapters,
    weights=top_weights,
    adapter_name="merged",
    combination_type="linear"
)
model.set_adapter("merged")

# Single forward pass with blended expertise
response = model.generate(query)
```

For a query like *"Write a Python function that solves the Fibonacci sequence mathematically"*, the gate fires high on both `code` and `math`. The merged adapter carries expertise from both — the base model alone would produce a worse answer than either individual adapter, and a hard-routed system would pick only one.

---

## Evaluation

| Metric | Description | Target |
|---|---|---|
| Gate accuracy | % of queries routed to correct adapter on held-out test set | > 90% |
| Response quality delta | Base model vs routed adapter on domain benchmarks (HumanEval, MATH eval) | measurable positive delta |
| Routing latency | Gate inference time | < 10 ms |
| Adapter merge time | `add_weighted_adapter()` call | < 100 ms |

The key evaluation is a side-by-side comparison: the exact same query answered by the base model alone versus the soft-routed adapter blend. The difference in response specificity and accuracy constitutes the "quality delta".

---

## Project Structure

```
adaproute/
├── data/
│   └── prepare_datasets.py       # Download and label task datasets
├── gate/
│   ├── train_gate.py             # Fine-tune DistilBERT classifier
│   └── evaluate_gate.py          # Accuracy, confusion matrix
├── adapters/
│   ├── train_adapter.py          # SFT one LoRA adapter (pass --domain flag)
│   └── merge_adapters.py         # add_weighted_adapter() inference wrapper
├── eval/
│   └── benchmark.py              # Quality delta across domains
├── demo/
│   └── app.py                    # Gradio demo
├── notebooks/
│   └── AdaptRoute.ipynb          # End-to-end walkthrough
└── README.md
```

---

## Setup

```bash
pip install transformers peft trl bitsandbytes datasets gradio wandb
```

```bash
# 1. Prepare datasets
python data/prepare_datasets.py

# 2. Train the gating network
python gate/train_gate.py

# 3. Train LoRA adapters (run once per domain)
python adapters/train_adapter.py --domain code
python adapters/train_adapter.py --domain math
python adapters/train_adapter.py --domain qa
python adapters/train_adapter.py --domain medical

# 4. Run the demo
python demo/app.py
```

---

## Connection to MoE

In a standard MoE transformer, the gating function and the experts are trained jointly end-to-end. The gate learns to route tokens to the experts that minimize loss — but this requires all experts to exist inside the model at training time, on the same hardware, with a specialized distributed training setup.

AdaptRoute decouples this:

| | MoE (standard) | AdaptRoute |
|---|---|---|
| Gate training | Joint with experts | Independent classifier |
| Expert training | Joint, end-to-end | Independent SFT per adapter |
| Routing | Soft (differentiable) | Soft (post-hoc weight blend) |
| Hardware | Multi-GPU required | Single A100 sufficient |
| Expert swap cost | Cannot swap at inference | Free — adapters are files |

The tradeoff is that AdaptRoute's gate is not trained to minimize the same loss as the experts — it optimizes task classification accuracy, not generation quality. This is the research gap worth acknowledging, and measuring, in the eval.

---

## References

- [PEFT library — `add_weighted_adapter`](https://huggingface.co/docs/peft)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Mixtral of Experts (MoE reference)](https://arxiv.org/abs/2401.04088)
- [TRL — SFTTrainer](https://huggingface.co/docs/trl)
- [Qwen2.5 model family](https://huggingface.co/Qwen)
- [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)
- [lighteval/MATH](https://huggingface.co/datasets/lighteval/MATH)
- [rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2)
- [lavita/ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)