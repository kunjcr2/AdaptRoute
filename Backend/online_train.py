"""
AdaptRoute - Continual Learning Loop (H200, Python 3.11.15)
============================================================
Offline retraining half of AdaptRoute. Runs GRPO on logged queries to
continuously update per-domain LoRA adapters.

Key design decisions:
  - GRPO needs prompts + reward fn only. No reference answers required.
  - Reward is rule-based (no BERTScore, no reference model):
      reward = 0.40 * length_score    (encourages concise, complete answers)
             + 0.30 * termination     (penalizes truncation hard)
             + 0.20 * repetition      (penalizes n-gram loops)
             + 0.10 * structure       (rewards coherent formatting)
    Mapped from [0,1] -> [-1,1] for GRPO.
  - No BERTScorer means no reference-answer lock-in, no extra GPU model,
    and the reward ceiling is not capped by past response quality.

Install (tested on H200, CUDA 12.4, torch 2.5+):
  pip install "transformers>=4.47,<4.50" "trl>=0.16,<0.20" \\
              "peft>=0.14" "accelerate>=1.2" "datasets>=3.0" \\
              "huggingface_hub>=0.27" "wandb>=0.18"

Python: 3.11.15
"""

from __future__ import annotations

import gc
import json
import os
import random
import re
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import wandb
from datasets import Dataset
from huggingface_hub import login
from peft import PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API")

HF_USERNAME = "kunjcr2"
VERSION = "v4"

BASE_MODEL_ID = "google/gemma-3-1b-it"

ADAPTER_REPOS: dict[str, str] = {
    "code":    f"{HF_USERNAME}/code-adaptroute-{VERSION}",
    "math":    f"{HF_USERNAME}/math-adaptroute-{VERSION}",
    "medical": f"{HF_USERNAME}/medical-adaptroute-{VERSION}",
}

LOG_FILE = Path("./query_log.jsonl")
OUTPUT_DIR = Path("./continual_learning")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
LOG_STEPS = 10
MIN_SAMPLES_TO_RETRAIN = 5

# Gemma-3 end-of-turn string. Generation stops here, so completions terminate
# naturally instead of getting length-capped.
GEMMA_EOT = "<end_of_turn>"

# Target completion budget. 512 tokens is enough for most code/math/medical
# answers. If your logged answers are consistently longer, raise this.
MAX_COMPLETION_LENGTH = 512

# Ideal length for the length_score reward. Answers at or below this get
# full credit; longer answers decay linearly.
IDEAL_COMPLETION_TOKENS = 384

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("[auth] HuggingFace login OK")

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)


# ---------------------------------------------------------------------------
# STEP 1 - LOAD QUERY LOG
# ---------------------------------------------------------------------------
def load_query_log() -> list[dict]:
    """
    Load logged queries from query_log.jsonl. Each line should be:
        {"question": str, "domain": str, "timestamp": float, ...}

    Note: 'answer' field is NOT required for GRPO. If present, it's ignored
    during training - GRPO generates fresh completions and scores them via
    the reward function.
    """
    if not LOG_FILE.exists():
        print(f"[log] {LOG_FILE} not found - nothing to train on")
        return []

    records: list[dict] = []
    bad_lines = 0
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "question" in rec and "domain" in rec:
                    records.append(rec)
                else:
                    bad_lines += 1
            except json.JSONDecodeError:
                bad_lines += 1

    print(f"[log] Loaded {len(records)} records from {LOG_FILE}"
          + (f" ({bad_lines} malformed lines skipped)" if bad_lines else ""))
    return records


# ---------------------------------------------------------------------------
# STEP 2 - REWARD FUNCTION (rule-based, no reference answers)
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")


def _tokenize_words(text: str) -> list[str]:
    return _WS_RE.split(text.strip())


def repetition_fraction(text: str, ngram: int = 4) -> float:
    """Fraction of n-grams that are duplicates. 0.0 = no repetition."""
    words = _tokenize_words(text)
    if len(words) < ngram:
        return 0.0
    grams = [tuple(words[i:i + ngram]) for i in range(len(words) - ngram + 1)]
    if not grams:
        return 0.0
    return (len(grams) - len(set(grams))) / len(grams)


def length_score(text: str, ideal_max: int = IDEAL_COMPLETION_TOKENS) -> float:
    """
    1.0 at <= ideal_max whitespace tokens, linear decay to 0.0 at 2*ideal_max,
    stays at 0.0 beyond. Penalizes both empty strings and overly long answers.
    """
    n = len(_tokenize_words(text))
    if n == 0:
        return 0.0
    if n <= ideal_max:
        # slight bonus for having *some* content: anything under ~20 tokens
        # is probably a non-answer ("I don't know", "See above", etc.)
        if n < 20:
            return n / 20.0
        return 1.0
    return max(0.0, 1.0 - (n - ideal_max) / ideal_max)


def termination_score(text: str) -> float:
    """
    1.0 if the completion looks like it ended on its own, 0.0 if truncated.
    Checks for sentence-ending punctuation or Gemma's end-of-turn marker.
    """
    stripped = text.rstrip()
    if not stripped:
        return 0.0
    if stripped.endswith(GEMMA_EOT):
        return 1.0
    # Cheap heuristic: does it end on punctuation that implies completion?
    if stripped[-1] in ".!?}])\"'":
        return 1.0
    # Ends mid-word or mid-sentence -> truncated
    return 0.0


def structure_score(text: str) -> float:
    """
    Small reward for coherent structure: sentences exist, not one giant
    run-on. Returns value in [0, 1].
    """
    n_words = len(_tokenize_words(text))
    if n_words < 10:
        return 0.0
    # Count sentence-ish breaks
    n_sentences = sum(text.count(c) for c in ".!?")
    if n_sentences == 0:
        return 0.2  # one long paragraph, not ideal but not zero
    avg_sentence_len = n_words / max(n_sentences, 1)
    # Sweet spot: 10-30 words per sentence
    if 8 <= avg_sentence_len <= 35:
        return 1.0
    if avg_sentence_len < 8:
        return 0.6  # too choppy
    return max(0.3, 1.0 - (avg_sentence_len - 35) / 50.0)  # too long-winded


def compute_scalar_reward(text: str) -> float:
    """
    Returns reward in [-1, 1]. Combines four rule-based signals.
    No reference answer needed.
    """
    len_s = length_score(text)
    term_s = termination_score(text)
    rep_s = 1.0 - repetition_fraction(text)
    struct_s = structure_score(text)

    raw = (
        0.40 * len_s
        + 0.30 * term_s
        + 0.20 * rep_s
        + 0.10 * struct_s
    )
    return raw * 2.0 - 1.0


def grpo_reward_fn(completions: list, **_kwargs) -> list[float]:
    """
    Called by GRPOTrainer. `completions` is a list of either:
      - list of chat dicts: [{"role": "assistant", "content": "..."}]
      - plain strings (if using completion-only format)
    """
    texts: list[str] = []
    for c in completions:
        if isinstance(c, list) and c:
            texts.append(c[-1].get("content", "") if isinstance(c[-1], dict) else str(c[-1]))
        elif isinstance(c, dict):
            texts.append(c.get("content", ""))
        else:
            texts.append(str(c) if c is not None else "")

    return [compute_scalar_reward(t) for t in texts]


# ---------------------------------------------------------------------------
# STEP 3 - GRPO RETRAINING PER DOMAIN
# ---------------------------------------------------------------------------
def retrain_adapter(domain: str, domain_records: list[dict]) -> None:
    """
    Continue training an existing LoRA adapter on logged prompts.
    The existing adapter is loaded with is_trainable=True; GRPO updates it
    in place, then pushes back to the same HF repo.
    """
    print(f"\n{'=' * 60}")
    print(f"CONTINUOUS TRAINING - {domain.upper()}")
    print(f"{'=' * 60}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for generation in GRPO

    # Sanity: confirm Gemma-3 EOT is the turn-terminator. Generation config
    # should stop on this, not on generic <eos>.
    eot_ids = tokenizer.encode(GEMMA_EOT, add_special_tokens=False)
    print(f"[{domain}] EOT token '{GEMMA_EOT}' -> ids {eot_ids}")

    # Base model. bf16 on H200 is the right default.
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        token=HF_TOKEN if HF_TOKEN else None,
    )

    # Load the existing adapter as trainable so GRPO updates it in place.
    print(f"[{domain}] Loading existing adapter from {ADAPTER_REPOS[domain]}")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_REPOS[domain],
        adapter_name=domain,
        is_trainable=True,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    model.print_trainable_parameters()

    # Dataset: prompts only. GRPO generates completions itself.
    prompts = [[{"role": "user", "content": rec["question"]}] for rec in domain_records]
    grpo_dataset = Dataset.from_dict({"prompt": prompts})

    # GRPO config - tuned for continued training on H200
    run_name = f"adaptroute-{domain}-{VERSION}-{int(time.time())}"
    grpo_cfg = GRPOConfig(
        output_dir=str(OUTPUT_DIR / f"grpo-{domain}"),
        run_name=run_name,

        # Training schedule
        num_train_epochs=1,
        per_device_train_batch_size=4,       # H200 has 141GB, can handle more
        gradient_accumulation_steps=2,
        learning_rate=5e-6,                  # live range for LoRA continued training
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,

        # Precision - H200 supports both
        bf16=True,
        tf32=True,

        # GRPO specifics
        beta=0.04,                           # KL anchor, prevents drift & loss=0 degeneracy
        num_generations=4,                   # group size for advantage computation
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=0.9,
        top_p=0.95,

        # Logging / checkpointing
        logging_steps=LOG_STEPS,
        save_strategy="no",                  # we push to HF at the end; skip disk checkpoints
        report_to="wandb" if WANDB_API_KEY else "none",
        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,                         # adapter already attached
        reward_funcs=grpo_reward_fn,
        args=grpo_cfg,
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
        # no peft_config - model already wraps LoRA
    )

    trainer.train()

    # Push updated adapter back to same repo
    print(f"[{domain}] Pushing updated adapter -> {ADAPTER_REPOS[domain]}")
    trainer.model.push_to_hub(ADAPTER_REPOS[domain], token=HF_TOKEN)
    tokenizer.push_to_hub(ADAPTER_REPOS[domain], token=HF_TOKEN)

    if WANDB_API_KEY:
        wandb.finish()

    del model, base_model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# STEP 4 - INFERENCE-SIDE LOGGING HELPER
# ---------------------------------------------------------------------------
def log_query(question: str, model_response: str, domain: str) -> None:
    """
    Append a query + response to query_log.jsonl. Call this from
    pipeline_v4.py after each generation.

    Note: model_response is logged for analytics/inspection, but GRPO
    does NOT use it as a reference - it generates fresh completions
    during training.
    """
    record = {
        "question": question,
        "response": model_response,          # kept for inspection only
        "domain": domain,
        "timestamp": time.time(),
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
def run_retraining_loop(
    stop_event: Any = None,
    on_phase: Callable[[str, str | None], None] | None = None,
) -> None:
    """
    Offline retraining pipeline:
      1. Load query logs
      2. Group by domain
      3. For each domain with >= MIN_SAMPLES_TO_RETRAIN -> run GRPO -> push

    Args:
        stop_event: threading.Event - when set, stops between domains
        on_phase: callback(phase, domain) for external progress tracking.
                  Phases: loading, training, done_domain, skipped,
                          complete, stopped
    """
    def _phase(phase: str, domain: str | None = None) -> None:
        if on_phase:
            try:
                on_phase(phase, domain)
            except Exception:
                pass

    t0 = time.time()
    print("\n" + "#" * 60)
    print("  AdaptRoute - Continual Learning Loop")
    print("#" * 60 + "\n")

    _phase("loading")
    all_records = load_query_log()

    if not all_records:
        print("[main] No records to train on. Exiting.")
        _phase("complete")
        return

    # Group by domain
    for domain in ADAPTER_REPOS.keys():
        if stop_event and stop_event.is_set():
            print("[stop] Halting on user request")
            _phase("stopped")
            return

        domain_records = [r for r in all_records if r.get("domain") == domain]

        if len(domain_records) < MIN_SAMPLES_TO_RETRAIN:
            print(f"\n[skip] {domain}: {len(domain_records)} samples "
                  f"(need {MIN_SAMPLES_TO_RETRAIN})")
            _phase("skipped", domain)
            continue

        print(f"\n[{domain}] {len(domain_records)} samples queued")
        _phase("training", domain)
        retrain_adapter(domain, domain_records)
        _phase("done_domain", domain)
        print(f"[timer] After {domain}: {(time.time() - t0) / 3600:.2f} hrs")

    total = (time.time() - t0) / 3600
    print("\n" + "#" * 60)
    print(f"  Continual Learning Complete in {total:.2f} hrs")
    for d, r in ADAPTER_REPOS.items():
        print(f"    {d:<8} -> {r}")
    print("#" * 60 + "\n")
    _phase("complete")


if __name__ == "__main__":
    run_retraining_loop()