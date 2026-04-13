# AdaptRoute — Interactive Notebooks

This directory contains the core training and integration logic for the AdaptRoute system, designed for high-performance routing on edge devices.

## Hub Models
The primary gating model is hosted on Hugging Face:
- **Firewall:** [kunjcr2/bert-lora](https://huggingface.co/kunjcr2/bert-lora)
- **Gating Network:** [kunjcr2/gating-bert-adaptroute](https://huggingface.co/kunjcr2/gating-bert-adaptroute)

## Notebooks

### 1. [Firewall.ipynb](./Firewall.ipynb)
- For more info on the firewall model, visit [llm-firewall](https://github.com/kunjcr2/llms-from-scratch/tree/main/models/llm-firewall)

**Purpose:** Sets up the first-line security layer.
- **Task:** Binary classification (Clear vs. Adversarial/OOD).
- **Function:** Acts as a gateway that must "clear" a query before it is passed to the gating network.

### 2. [train_gate_adaptroute.ipynb](./train_gate_adaptroute.ipynb)
**Purpose:** Fine-tunes a DistilBERT-based gating network using LoRA.
- **Task:** 4-class classification (`code`, `math`, `qa`, `medical`).
- **Optimization:** Uses class-balanced training and weighted loss to optimize for high-accuracy routing across specialized domains.
- **Efficiency:** Leverages PEFT/LoRA to keep the trainable parameter count low (~1.5% of total params).

<!-- ### 3. [AdaptRoute.ipynb](file:///c:/Users/kunjs/Downloads/Projects/AdaptRoute/Notebooks/AdaptRoute.ipynb)
**Purpose:** The end-to-end system walkthrough.
- **Workflow:** demonstrates query input → firewall → clears → gating → soft adapter merging → base model inference.
- **Key Feature:** Showcases the complete pipeline: security filtering, dynamic task routing, and soft blending of expertise. -->

---

## Environment Setup
All notebooks are optimized for **Google Colab (A100)** but can be run locally with:
```bash
pip install transformers peft bitsandbytes datasets trl wandb
```