# ============================================================
# AdaptRoute — LoRA Adapter Training (H200 Edition)
# ============================================================
# H200-specific upgrades vs A100 version:
#   • Full bfloat16 — no QLoRA (141 GB HBM3e fits the base model + all grads comfortably)
#   • Flash Attention 2 for training kernels
#   • BATCH_SIZE 8 → 64, GRAD_ACCUM 4 → 1 (true batch of 64, not fake accumulation)
#   • LORA_R upgraded: code/math 32→64, qa/medical 16→32
#   • MAX_LENGTH upgraded: code/math 1024→2048
#   • Packing=True — H200 memory makes it safe, halves steps for same data
#   • Dataset upgrades per domain (see table below):
#
#   Domain  | Old dataset                          | New dataset
#   --------|--------------------------------------|------------------------------------------
#   code    | iamtarun/python_code_instructions    | m-a-p/CodeFeedback-Filtered-Instruction
#           |   18k simple alpaca-style            |   74k GPT-4 complexity-filtered, multi-turn
#   math    | DigitalLearningGmbH/MATH-lighteval   | AI-MO/NuminaMath-CoT
#           |   ~7.5k problems                     |   860k competition math w/ CoT chains
#   qa      | rajpurkar/squad                      | hotpotqa/hotpot_qa (distractor)
#           |   extractive span-finding            |   113k multi-hop reasoning questions
#   medical | lavita/ChatDoctor-HealthCareMagic    | FreedomIntelligence/medical-o1-reasoning-SFT
#           |   chat-style health advice           |   GPT-4o verified CoT reasoning chains
# ============================================================

# ─── USER CONFIG ──────────────────────────────────────────────────────────────
HF_USERNAME   = "kunjcr2"
WANDB_PROJECT = "adaptroute-adapters-v4"

BASE_MODEL    = "Qwen/Qwen2.5-1.5B"
OUTPUT_ROOT   = "./adapters"
SEED          = 42

# H200: no quantization, real bf16 throughout
USE_QLORA         = False    # H200 has 141 GB — no need for 4-bit

# LoRA — bigger ranks now that we have memory
LORA_DROPOUT         = 0.05
LORA_TARGET_MODULES  = "all-linear"

# SFT defaults — tuned for H200 single GPU
BATCH_SIZE   = 64     # was 8 on A100 w/ GRAD_ACCUM=4 (effective=32); now true batch=64
GRAD_ACCUM   = 1      # no accumulation needed
NUM_EPOCHS   = 3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.3
WARMUP_RATIO  = 0.03
LOGGING_STEPS = 10
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "no"
PACKING       = True   # safe on H200, ~2x throughput vs packing=False
# ──────────────────────────────────────────────────────────────────────────────

ADAPTERS = [
    {
        "name": "code",
        # CodeFeedback-Filtered-Instruction: 74k instructions filtered from 287k
        # by Qwen-72B-Chat for complexity score ≥4/5. Multi-turn, multi-language.
        # Massive upgrade over 18k simple alpaca-style Python snippets.
        "hf_dataset": "m-a-p/CodeFeedback-Filtered-Instruction",
        "hf_config": None,
        "hf_split": "train",
        "stream": False,
        "num_epochs": 3,
        "max_length": 2048,   # code needs longer context
        "n_samples": 60_000,  # was 18k; H200 handles 60k in same wall time
        "learning_rate": 8e-5,
        "lora_r": 64,         # was 32
        "lora_alpha": 128,    # keep 2x r
    },
    {
        "name": "math",
        # NuminaMath-CoT: 860k competition math problems (AMC→Olympiad)
        # each with templated Chain-of-Thought. The dataset that won AIMO 2024.
        # vs MATH-lighteval which has ~7.5k problems with no CoT structure.
        "hf_dataset": "AI-MO/NuminaMath-CoT",
        "hf_config": None,
        "hf_split": "train",
        "stream": False,
        "num_epochs": 2,      # huge dataset — 2 epochs is plenty
        "max_length": 2048,
        "n_samples": 100_000, # was 20k
        "learning_rate": 6e-5,
        "lora_r": 64,
        "lora_alpha": 128,
    },
    {
        "name": "qa",
        # HotpotQA (distractor setting): 90k multi-hop questions requiring
        # reasoning over 2 Wikipedia paragraphs. Much harder than SQuAD
        # (span extraction). Forces the model to synthesize across documents.
        "hf_dataset": "hotpotqa/hotpot_qa",
        "hf_config": "distractor",
        "hf_split": "train",
        "stream": False,
        "num_epochs": 3,
        "max_length": 1024,
        "n_samples": 80_000,  # was 20k
        "learning_rate": 8e-5,
        "lora_r": 32,         # was 16
        "lora_alpha": 64,
    },
    {
        "name": "medical",
        # medical-o1-reasoning-SFT: GPT-4o built CoT chains for verifiable
        # medical problems (board-exam style). Each sample has Question,
        # Complex_CoT (reasoning trace), and Response (final answer).
        # vs ChatDoctor which is casual health advice with no reasoning structure.
        "hf_dataset": "FreedomIntelligence/medical-o1-reasoning-SFT",
        "hf_config": "en",
        "hf_split": "train",
        "stream": False,
        "num_epochs": 3,
        "max_length": 2048,   # CoT chains are long
        "n_samples": 40_000,  # was 16k
        "learning_rate": 5e-5,
        "lora_r": 32,
        "lora_alpha": 64,
    },
]

## 2. Authentication (unchanged)
from google.colab import userdata
from huggingface_hub import login

HF_TOKEN = userdata.get("HF_TOKEN")
login(token=HF_TOKEN, add_to_git_credential=False)
print("✓ HuggingFace login OK")
if WANDB_PROJECT:
    import wandb
    wandb.login(key=userdata.get("WANDB_API"))
    print("✓ W&B ready")

## 3. Dataset Formatters

def make_prompt_completion(user_text: str, assistant_text: str):
    prompt = f"<|im_start|>user\n{user_text.strip()}<|im_end|>\n<|im_start|>assistant\n"
    completion = f"{assistant_text.strip()}<|im_end|>"
    return {"prompt": prompt, "completion": completion}

def format_code(record):
    # CodeFeedback schema: 'query' + 'answer'
    query  = record.get("query", "").strip()
    answer = record.get("answer", "").strip()
    if not query or not answer:
        return None
    return make_prompt_completion(query, answer)

def format_math(record):
    # NuminaMath-CoT schema: 'problem' + 'solution'
    problem  = record.get("problem", "").strip()
    solution = record.get("solution", "").strip()
    if not problem or not solution:
        return None
    # Prepend CoT instruction so the model learns to show its work
    user_turn = (
        "Solve the following math problem step by step. "
        "Show all reasoning clearly.\n\n"
        + problem
    )
    return make_prompt_completion(user_turn, solution)

def format_qa(record):
    # HotpotQA distractor schema: 'question', 'answer', 'context' (dict of titles+sentences)
    question = record.get("question", "").strip()
    answer   = record.get("answer", "").strip()
    context  = record.get("context", {})
    if not question or not answer:
        return None

    # Build a readable context block from the distractor paragraphs
    titles    = context.get("title", [])
    sentences = context.get("sentences", [])
    passages  = []
    for title, sents in zip(titles, sentences):
        passages.append(f"{title}: {''.join(sents)}")
    context_str = "\n\n".join(passages[:4])   # top 4 paragraphs fits in max_length

    user_turn = (
        "Read the passages and answer the question. "
        "The question requires reasoning over multiple passages.\n\n"
        f"Passages:\n{context_str}\n\nQuestion: {question}"
    )
    return make_prompt_completion(user_turn, answer)

def format_medical(record):
    # medical-o1-reasoning-SFT schema: 'Question', 'Complex_CoT', 'Response'
    question = record.get("Question", "").strip()
    cot      = record.get("Complex_CoT", "").strip()
    response = record.get("Response", "").strip()
    if not question or not response:
        return None

    # Combine CoT + Response as the assistant turn so the model learns to reason
    assistant_turn = f"{cot}\n\n{response}" if cot else response

    user_turn = (
        "You are a clinical reasoning assistant. "
        "Think through the problem carefully before giving your answer.\n\n"
        + question
    )
    return make_prompt_completion(user_turn, assistant_turn)

FORMATTERS = {
    "code":    format_code,
    "math":    format_math,
    "qa":      format_qa,
    "medical": format_medical,
}

print("✓ Formatters defined")

## 4. Load Base Model — Full bfloat16, no quantization

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"Loading {BASE_MODEL} in bfloat16 (no quantization — H200 has 141 GB)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",   # FA2 during training
    device_map="auto",
    trust_remote_code=True,
)
base_model.config.use_cache = False   # must be False during training w/ grad checkpointing

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

total_params = sum(p.numel() for p in base_model.parameters())
print(f"✓ Loaded | Total params: {total_params/1e9:.2f}B | dtype: {next(base_model.parameters()).dtype}")

## 5. Training Loop

import gc
import random
from pathlib import Path

import wandb
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi

random.seed(SEED)
api = HfApi()


def load_and_format(adapter_cfg: dict, formatter, n: int) -> Dataset:
    hf_ds  = adapter_cfg["hf_dataset"]
    cfg    = adapter_cfg["hf_config"]
    split  = adapter_cfg["hf_split"]
    kwargs = {"split": split, "trust_remote_code": True}
    if cfg:
        kwargs["name"] = cfg
    raw     = load_dataset(hf_ds, **kwargs)
    records = list(raw)
    random.shuffle(records)

    rows = []
    for rec in records:
        formatted = formatter(rec)
        if not formatted or len(formatted["completion"]) < 8:
            continue
        rows.append(formatted)
        if len(rows) >= n:
            break

    print(f"  Loaded {len(rows)} samples from {hf_ds}")
    return Dataset.from_list(rows)


def train_adapter(adapter_cfg: dict) -> None:
    name       = adapter_cfg["name"]
    repo_id    = f"{HF_USERNAME}/{name}-adaptroute-v4"
    output_dir = f"{OUTPUT_ROOT}/{name}"
    formatter  = FORMATTERS[name]

    adapter_epochs = adapter_cfg.get("num_epochs", NUM_EPOCHS)
    adapter_maxlen = adapter_cfg.get("max_length", 1024)
    n_samples      = adapter_cfg.get("n_samples", 20_000)
    lora_r         = adapter_cfg.get("lora_r", 32)
    lora_alpha     = adapter_cfg.get("lora_alpha", 64)

    print(f"\n{'='*65}")
    print(f"  TRAINING: {name.upper()}  →  {repo_id}")
    print(f"  r={lora_r} α={lora_alpha} | epochs={adapter_epochs} "
          f"| maxlen={adapter_maxlen} | n={n_samples}")
    print(f"{'='*65}")

    dataset = load_and_format(adapter_cfg, formatter, n_samples)

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

    if WANDB_PROJECT:
        wandb.init(project=WANDB_PROJECT, name=f"{name}-v4", reinit=True)

    sft_config = SFTConfig(
        output_dir                         = output_dir,
        num_train_epochs                   = adapter_epochs,
        per_device_train_batch_size        = BATCH_SIZE,
        gradient_accumulation_steps        = GRAD_ACCUM,
        learning_rate                      = adapter_cfg.get("learning_rate", 8e-5),
        warmup_ratio                       = WARMUP_RATIO,
        lr_scheduler_type                  = "cosine",
        bf16                               = True,
        fp16                               = False,
        logging_steps                      = LOGGING_STEPS,
        save_strategy                      = SAVE_STRATEGY,
        eval_strategy                      = EVAL_STRATEGY,
        report_to                          = "wandb" if WANDB_PROJECT else "none",
        max_length                         = adapter_maxlen,
        packing                            = PACKING,
        seed                               = SEED,
        weight_decay                       = WEIGHT_DECAY,
        max_grad_norm                      = MAX_GRAD_NORM,
        gradient_checkpointing             = True,
        gradient_checkpointing_kwargs      = {"use_reentrant": False},
        # H200: dataloader workers saturate the CPU prefetch buffer
        dataloader_num_workers             = 4,
        dataloader_pin_memory              = True,
    )

    trainer = SFTTrainer(
        model            = model,
        args             = sft_config,
        train_dataset    = dataset,
        processing_class = tokenizer,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  ✓ Saved → {output_dir}")

    # Model card
    card = f"""---
base_model: {BASE_MODEL}
library_name: peft
license: apache-2.0
tags: [lora, peft, adaptroute, {name}]
---
# {name}-adaptroute-v4

LoRA adapter for the **{name}** domain in AdaptRoute (H200 edition).

| Setting | Value |
|---------|-------|
| Base | `{BASE_MODEL}` bf16, no quantization |
| r / α | {lora_r} / {lora_alpha} |
| Dataset | `{adapter_cfg['hf_dataset']}` |
| Samples | {n_samples} |
| Epochs | {adapter_epochs} |
| Max length | {adapter_maxlen} |
| Packing | {PACKING} |
| Batch | {BATCH_SIZE} (no grad accum) |
"""
    Path(f"{output_dir}/README.md").write_text(card)

    api.create_repo(repo_id=repo_id, exist_ok=True, token=HF_TOKEN)
    api.upload_folder(
        folder_path    = output_dir,
        repo_id        = repo_id,
        token          = HF_TOKEN,
        commit_message = f"Add {name}-adaptroute-v4 (H200)",
    )
    print(f"  ✓ Pushed → https://huggingface.co/{repo_id}")

    if WANDB_PROJECT:
        wandb.finish()
    del model, trainer, dataset
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  ✓ VRAM cleared\n")


## 6. Run All 4 Adapters
for adapter_cfg in ADAPTERS:
    train_adapter(adapter_cfg)

print("\n" + "="*65)
print("ALL ADAPTERS TRAINED AND PUSHED (v4 / H200)")
for a in ADAPTERS:
    print(f"  https://huggingface.co/{HF_USERNAME}/{a['name']}-adaptroute-v4")
print("="*65)

## 7. Verification (updated prompts + repo names)
from peft import PeftModel

VERIFY_PROMPTS = {
    "code": (
        "Write a Python function that, given a list of integers, "
        "returns all pairs whose sum equals a target value. "
        "Use a hash set, not nested loops."
    ),
    "math": (
        "Solve step by step: A geometric sequence has first term 3 "
        "and common ratio 2. Find the sum of the first 10 terms."
    ),
    "qa": (
        "Passages:\n"
        "Alan Turing: Alan Turing was an English mathematician and computer scientist.\n"
        "Turing Award: The Turing Award is given annually by the ACM to individuals "
        "of lasting technical importance in computing.\n\n"
        "Question: What is the connection between the person the Turing Award is named "
        "after and the field of computing?"
    ),
    "medical": (
        "A 62-year-old male smoker presents with a persistent cough, "
        "haemoptysis, and 8 kg weight loss over 3 months. Chest X-ray "
        "shows a right hilar mass. What is the most likely diagnosis "
        "and what are the immediate next steps?"
    ),
}

print("Verifying v4 adapters...\n")
for a in ADAPTERS:
    name    = a["name"]
    repo_id = f"{HF_USERNAME}/{name}-adaptroute-v4"
    prompt  = VERIFY_PROMPTS[name]
    print(f"── {name.upper()} ──")
    try:
        peft_model = PeftModel.from_pretrained(base_model, repo_id)
        peft_model.eval()
        enc = tokenizer(
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            return_tensors="pt",
        ).to(base_model.device)
        with torch.no_grad():
            out = peft_model.generate(
                **enc,
                max_new_tokens = 300,
                do_sample      = False,
                temperature    = None,
                top_p          = None,
            )
        response = tokenizer.decode(
            out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
        )
        print(f"Response:\n{response}\n✓ OK\n")
        del peft_model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ FAILED: {e}\n")