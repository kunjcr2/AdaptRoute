# -*- coding: utf-8 -*-
# AdaptRoute — Adapter Training v4
# Target: H200 Linux VM  |  Python 3.11  |  bfloat16, no quantization
#
# ── Before running, install deps ONCE in your venv: ──────────────────────────
#   pip install -U transformers trl peft datasets accelerate bitsandbytes \
#               huggingface_hub wandb liger-kernel
# ─────────────────────────────────────────────────────────────────────────────

# ── Cell 2: Imports ───────────────────────────────────────────
import gc
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
import wandb
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Python 3.9 compatibility — use Optional/List/Dict from typing
from typing import Dict, List

print(f"Python : {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA   : {torch.version.cuda}")
print(f"Device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── Cell 3: Configuration ─────────────────────────────────────
HF_USERNAME   = "kunjcr2"
WANDB_PROJECT = "adaptroute-adapters-v4"

BASE_MODEL  = "Qwen/Qwen2.5-1.5B"
OUTPUT_ROOT = "./adapters"
SEED        = 42

# H200 — full bfloat16, no quantization
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = "all-linear"

BATCH_SIZE    = 64
GRAD_ACCUM    = 1
NUM_EPOCHS    = 3
WEIGHT_DECAY  = 0.01
MAX_GRAD_NORM = 0.3
WARMUP_RATIO  = 0.03
LOGGING_STEPS = 10
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "no"
PACKING       = True

ADAPTERS = [
    {
        "name":          "code",
        "hf_dataset":    "m-a-p/CodeFeedback-Filtered-Instruction",
        "hf_config":     None,
        "hf_split":      "train",
        "num_epochs":    3,
        "max_length":    2048,
        "n_samples":     60_000,
        "learning_rate": 8e-5,
        "lora_r":        64,
        "lora_alpha":    128,
    },
    {
        "name":          "math",
        "hf_dataset":    "AI-MO/NuminaMath-CoT",
        "hf_config":     None,
        "hf_split":      "train",
        "num_epochs":    2,
        "max_length":    2048,
        "n_samples":     100_000,
        "learning_rate": 6e-5,
        "lora_r":        64,
        "lora_alpha":    128,
    },
    {
        "name":          "qa",
        "hf_dataset":    "hotpotqa/hotpot_qa",
        "hf_config":     "distractor",
        "hf_split":      "train",
        "num_epochs":    3,
        "max_length":    1024,
        "n_samples":     80_000,
        "learning_rate": 8e-5,
        "lora_r":        32,
        "lora_alpha":    64,
    },
    {
        "name":          "medical",
        "hf_dataset":    "FreedomIntelligence/medical-o1-reasoning-SFT",
        "hf_config":     "en",
        "hf_split":      "train",
        "num_epochs":    3,
        "max_length":    2048,
        "n_samples":     40_000,
        "learning_rate": 5e-5,
        "lora_r":        32,
        "lora_alpha":    64,
    },
]

print("✓ Config ready")

# ── Cell 4: Authentication ─────────────────────────────────────
HF_TOKEN = "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"

login(token=HF_TOKEN, add_to_git_credential=False)
print("✓ HuggingFace login OK")

if WANDB_PROJECT:
    wandb.login(key="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")
    print("✓ W&B ready")

from typing import Optional, Dict, List

# ── Cell 5: Dataset formatters ────────────────────────────────

def make_prompt_completion(
    user_text: str,
    assistant_text: str,
) -> Optional[Dict[str, str]]:
    prompt = (
        f"<|im_start|>user\n{user_text.strip()}"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )
    completion = f"{assistant_text.strip()}<|im_end|>"
    return {"prompt": prompt, "completion": completion}


def format_code(record: dict) -> Optional[Dict[str, str]]:
    query  = (record.get("query") or "").strip()
    answer = (record.get("answer") or "").strip()
    if not query or not answer:
        return None
    return make_prompt_completion(query, answer)


def format_math(record: dict) -> Optional[Dict[str, str]]:
    problem  = (record.get("problem") or "").strip()
    solution = (record.get("solution") or "").strip()
    if not problem or not solution:
        return None
    user_turn = (
        "Solve the following math problem step by step. "
        "Show all reasoning clearly.\n\n" + problem
    )
    return make_prompt_completion(user_turn, solution)


def format_qa(record: dict) -> Optional[Dict[str, str]]:
    question = (record.get("question") or "").strip()
    answer   = (record.get("answer")   or "").strip()
    context  = record.get("context") or {}
    if not question or not answer:
        return None

    titles    = context.get("title", [])
    sentences = context.get("sentences", [])
    passages  = []                               # type: List[str]
    for title, sents in zip(titles, sentences):
        passages.append(f"{title}: {''.join(sents)}")

    context_str = "\n\n".join(passages[:4])
    user_turn = (
        "Read the passages and answer the question. "
        "The question requires reasoning over multiple passages.\n\n"
        f"Passages:\n{context_str}\n\nQuestion: {question}"
    )
    return make_prompt_completion(user_turn, answer)


def format_medical(record: dict) -> Optional[Dict[str, str]]:
    question = (record.get("Question") or "").strip()
    cot      = (record.get("Complex_CoT") or "").strip()
    response = (record.get("Response") or "").strip()
    if not question or not response:
        return None

    assistant_turn = f"{cot}\n\n{response}" if cot else response
    user_turn = (
        "You are a clinical reasoning assistant. "
        "Think through the problem carefully before giving your answer.\n\n"
        + question
    )
    return make_prompt_completion(user_turn, assistant_turn)


FORMATTERS: Dict[str, object] = {
    "code":    format_code,
    "math":    format_math,
    "qa":      format_qa,
    "medical": format_medical,
}

# Sanity check
_dummy = {"problem": "What is 2+2?", "solution": "4"}
assert format_math(_dummy) is not None
print("✓ Formatters OK")

# ── Cell 6: Load base model (sdpa — no flash-attn required) ────

os.makedirs(OUTPUT_ROOT, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)

print(f"Loading {BASE_MODEL} — bfloat16, eager attention (avoiding SDPA CUDA bugs) ...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # Use eager (native) attention to avoid SDPA CUDA bugs
    device_map="auto",
    trust_remote_code=True,
)
base_model.config.use_cache = False   # required for gradient checkpointing

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

n_params = sum(p.numel() for p in base_model.parameters())
print(f"✓ Loaded | params: {n_params/1e9:.2f}B | dtype: {next(base_model.parameters()).dtype}")

# ── Cell 7: Data loader ────────────────────────────────────────

def load_and_format(
    adapter_cfg: dict,
    formatter,
    n: int,
) -> Dataset:
    hf_ds  = adapter_cfg["hf_dataset"]
    cfg    = adapter_cfg["hf_config"]
    split  = adapter_cfg["hf_split"]

    kwargs: Dict[str, object] = {"split": split, "trust_remote_code": True}
    if cfg:
        kwargs["name"] = cfg

    raw     = load_dataset(hf_ds, **kwargs)
    records = list(raw)
    random.shuffle(records)

    rows: List[Dict[str, str]] = []
    for rec in records:
        formatted = formatter(rec)
        if not formatted:
            continue
        if len(formatted["completion"]) < 8:
            continue
        rows.append(formatted)
        if len(rows) >= n:
            break

    print(f"  Collected {len(rows):,} samples from {hf_ds}")
    return Dataset.from_list(rows)

# ── Cell 8: Train one adapter ──────────────────────────────────

api = HfApi()

def train_adapter(adapter_cfg: dict) -> None:
    name       = adapter_cfg["name"]
    repo_id    = f"{HF_USERNAME}/{name}-adaptroute-v4"
    output_dir = f"{OUTPUT_ROOT}/{name}"
    formatter  = FORMATTERS[name]

    adapter_epochs = adapter_cfg.get("num_epochs", NUM_EPOCHS)
    adapter_maxlen = adapter_cfg.get("max_length", 1024)
    n_samples      = adapter_cfg.get("n_samples",  20_000)
    lora_r         = adapter_cfg.get("lora_r",     32)
    lora_alpha     = adapter_cfg.get("lora_alpha",  64)
    lr             = adapter_cfg.get("learning_rate", 8e-5)

    print(f"\n{'='*65}")
    print(f"  TRAINING : {name.upper()}  →  {repo_id}")
    print(f"  r={lora_r}  α={lora_alpha}  lr={lr}  "
          f"epochs={adapter_epochs}  maxlen={adapter_maxlen}  n={n_samples:,}")
    print(f"{'='*65}")

    # ── Data ──────────────────────────────────────────────────
    dataset = load_and_format(adapter_cfg, formatter, n_samples)

    # ── LoRA config ───────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = lora_r,
        lora_alpha     = lora_alpha,
        lora_dropout   = LORA_DROPOUT,
        target_modules = LORA_TARGET_MODULES,
        bias           = "none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # ── W&B ───────────────────────────────────────────────────
    if WANDB_PROJECT:
        wandb.init(
            project  = WANDB_PROJECT,
            name     = f"{name}-v4",
            reinit   = True,
            config   = {
                "domain":    name,
                "lora_r":    lora_r,
                "n_samples": n_samples,
                "lr":        lr,
            },
        )

    # ── SFT config ────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir                    = output_dir,
        num_train_epochs              = adapter_epochs,
        per_device_train_batch_size   = BATCH_SIZE,
        gradient_accumulation_steps   = GRAD_ACCUM,
        learning_rate                 = lr,
        warmup_ratio                  = WARMUP_RATIO,
        lr_scheduler_type             = "cosine",
        bf16                          = True,
        fp16                          = False,
        logging_steps                 = LOGGING_STEPS,
        save_strategy                 = SAVE_STRATEGY,
        eval_strategy                 = EVAL_STRATEGY,
        report_to                     = "wandb" if WANDB_PROJECT else "none",
        max_seq_length                = adapter_maxlen,
        packing                       = PACKING,
        seed                          = SEED,
        weight_decay                  = WEIGHT_DECAY,
        max_grad_norm                 = MAX_GRAD_NORM,
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": True},
        dataloader_num_workers        = 4,
        dataloader_pin_memory         = True,
        use_liger_kernel              = False,  # avoid transformers version mismatch
    )

    trainer = SFTTrainer(
        model            = model,
        args             = sft_config,
        train_dataset    = dataset,
        processing_class = tokenizer,
    )

    trainer.train()

    # ── Save + push ───────────────────────────────────────────
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  ✓ Adapter saved → {output_dir}")

    card = f"""---
base_model: {BASE_MODEL}
library_name: peft
license: apache-2.0
tags: [lora, peft, adaptroute, {name}]
---
# {name}-adaptroute-v4

| Setting | Value |
|---------|-------|
| Base model | `{BASE_MODEL}` bfloat16 sdpa |
| r / α | {lora_r} / {lora_alpha} |
| Dataset | `{adapter_cfg['hf_dataset']}` |
| Samples | {n_samples:,} |
| Epochs | {adapter_epochs} |
| Max length | {adapter_maxlen} |
| Batch size | {BATCH_SIZE} |
| Packing | {PACKING} |
"""
    Path(f"{output_dir}/README.md").write_text(card)

    api.create_repo(repo_id=repo_id, exist_ok=True, token=HF_TOKEN)
    api.upload_folder(
        folder_path    = output_dir,
        repo_id        = repo_id,
        token          = HF_TOKEN,
        commit_message = f"Add {name}-adaptroute-v4 (H200 sdpa)",
    )
    print(f"  ✓ Pushed → https://huggingface.co/{repo_id}")

    if WANDB_PROJECT:
        wandb.finish()

    # Free VRAM before next adapter
    del model, trainer, dataset
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  ✓ VRAM cleared\n")

# ── Cell 9: Run all 4 adapters ─────────────────────────────────
for adapter_cfg in ADAPTERS:
    train_adapter(adapter_cfg)

print("\n" + "="*65)
print("ALL ADAPTERS TRAINED AND PUSHED (v4 / H200 / sdpa)")
print("="*65)
for a in ADAPTERS:
    print(f"  https://huggingface.co/{HF_USERNAME}/{a['name']}-adaptroute-v4")