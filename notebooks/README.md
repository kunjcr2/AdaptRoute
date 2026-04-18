# AdaptRoute — Interactive Notebooks

This directory contains the core training and integration logic for the AdaptRoute system, designed for high-performance routing on edge devices.

## Hub Models
The primary routing models and expert adapters are hosted on Hugging Face:
- **Firewall:** [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) (Pre-trained)
- **Gating Network:** [kunjcr2/gating-bert-adaptroute-v4](https://huggingface.co/kunjcr2/gating-bert-adaptroute-v4)
- **Adapters (v4):**
  - [kunjcr2/code-adaptroute-v4](https://huggingface.co/kunjcr2/code-adaptroute-v4)
  - [kunjcr2/math-adaptroute-v4](https://huggingface.co/kunjcr2/math-adaptroute-v4)
  - [kunjcr2/medical-adaptroute-v4](https://huggingface.co/kunjcr2/medical-adaptroute-v4)

## Notebooks & Scripts

### 1. [Firewall.ipynb](./Firewall.ipynb)
- Uses the pre-trained [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) model.

**Purpose:** Sets up the first-line security layer.
- **Task:** Binary classification (SAFE vs. INJECTION).
- **Function:** Acts as a gateway that must "clear" a query before it is passed to the gating network.

### 2. [train_gate_adaptroute.ipynb](./train_gate_adaptroute.ipynb)
**Purpose:** Fine-tunes a DistilBERT-based gating network using LoRA.
- **Task:** 4-class classification (`code`, `math`, `medical`, `general`).
- **Efficiency:** Leverages PEFT/LoRA to keep the trainable parameter count low.

### 3. [Adapters_Training_v4.py](./Adapters_Training_v4.py)
**Purpose:** End-to-end multi-domain LoRA training pipeline.
- **Task:** Fine-tunes 3 separate LoRA adapters (`code`, `math`, `medical`) onto a frozen **Gemma-3-1B-It** model.
- **Optimization:** Uses LoRA rank (`r=16`) and alpha (`alpha=32`).

### 4. [online_train.py](../Backend/online_train.py)
**Purpose:** Continual Learning loop using GRPO.
- **Task:** Retrains adapters based on inference logs and reward scoring.

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