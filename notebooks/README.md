# AdaptRoute — Interactive Notebooks

This directory contains the core training and integration logic for the AdaptRoute system, designed for high-performance routing on edge devices.

## Hub Models
The primary routing models and expert adapters are hosted on Hugging Face:
- **Firewall:** [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) (Pre-trained)
- **Gating Network:** [kunjcr2/gating-bert-adaptroute](https://huggingface.co/kunjcr2/gating-bert-adaptroute)
- **Adapters (v2):**
  - [kunjcr2/code-adaptroute-v2](https://huggingface.co/kunjcr2/code-adaptroute-v2)
  - [kunjcr2/math-adaptroute-v2](https://huggingface.co/kunjcr2/math-adaptroute-v2)
  - [kunjcr2/qa-adaptroute-v2](https://huggingface.co/kunjcr2/qa-adaptroute-v2)
  - [kunjcr2/medical-adaptroute-v2](https://huggingface.co/kunjcr2/medical-adaptroute-v2)

## Notebooks

### 1. [Firewall.ipynb](./Firewall.ipynb)
- Uses the pre-trained [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) model.

**Purpose:** Sets up the first-line security layer.
- **Task:** Binary classification (SAFE vs. INJECTION).
- **Function:** Acts as a gateway that must "clear" a query before it is passed to the gating network. No custom training required — the model is loaded directly from Hugging Face.

### 2. [train_gate_adaptroute.ipynb](./train_gate_adaptroute.ipynb)
**Purpose:** Fine-tunes a DistilBERT-based gating network using LoRA.
- **Task:** 4-class classification (`code`, `math`, `qa`, `medical`).
- **Optimization:** Uses class-balanced training and weighted loss to optimize for high-accuracy routing across specialized domains.
- **Efficiency:** Leverages PEFT/LoRA to keep the trainable parameter count low (~1.5% of total params).

### 3. [Adapters_Training_v2.ipynb](./Adapters_Training_v2.ipynb)
**Purpose:** End-to-end multi-domain LoRA training pipeline.
- **Task:** Fine-tunes 4 separate LoRA adapters (`code`, `math`, `qa`, `medical`) onto a frozen Qwen2.5-1.5B model.
- **Optimization:** Streams data from Hugging Face directly, using an increased LoRA rank (`r=16`) and alpha (`alpha=32`).
- **Workflow:** Sequentially runs SFT and pushes `*-adaptroute-v2` models to the Hub in a single script.

<!-- ### 4. [AdaptRoute.ipynb](./AdaptRoute.ipynb)
**Purpose:** The end-to-end system walkthrough.
- **Workflow:** demonstrates query input → firewall → clears → gating → soft adapter merging → base model inference.
- **Key Feature:** Showcases the complete pipeline: security filtering, dynamic task routing, and soft blending of expertise. -->

---

## Environment Setup
All notebooks are optimized for **Google Colab (A100)** but can be run locally with:
```bash
pip install transformers peft bitsandbytes datasets trl wandb
```