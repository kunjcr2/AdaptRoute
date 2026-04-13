# AdaptRoute — Interactive Notebooks

This directory contains the core training and integration logic for the AdaptRoute system, designed for high-performance routing on edge devices.

## Hub Models
The primary gating model is hosted on Hugging Face:
- **Gating Network:** [kunjcr2/gating-bert-adaptroute](https://huggingface.co/kunjcr2/gating-bert-adaptroute)

## Notebooks

### 1. [train_gate_adaptroute.ipynb](file:///c:/Users/kunjs/Downloads/Projects/AdaptRoute/Notebooks/train_gate_adaptroute.ipynb)
**Purpose:** Fine-tunes a DistilBERT-based gating network using LoRA.
- **Task:** 4-class classification (`code`, `math`, `qa`, `medical`).
- **Optimization:** Uses class-balanced training and weighted loss to optimize for high-accuracy routing across specialized domains.
- **Efficiency:** Leverages PEFT/LoRA to keep the trainable parameter count low (~1.5% of total params).

### 2. [AdaptRoute.ipynb](file:///c:/Users/kunjs/Downloads/Projects/AdaptRoute/Notebooks/AdaptRoute.ipynb)
**Purpose:** The end-to-end system walkthrough.
- **Workflow:** demonstrates query input → gating → soft adapter merging → base model inference.
- **Key Feature:** Showcases how the router dynamically blends expertise from multiple adapters at runtime.

---

## Environment Setup
All notebooks are optimized for **Google Colab (A100)** but can be run locally with:
```bash
pip install transformers peft bitsandbytes datasets trl wandb
```