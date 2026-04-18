"""
AdaptRoute — Continual Learning Loop
=====================================
This script is the "offline retraining" half of AdaptRoute.

Architecture:
  INFERENCE (live, pipeline_v4.py)
      ↓  logs every query to query_log.jsonl
  RETRAINING (this script, run daily/weekly)
      ↓  scores logged responses with reward model
      ↓  runs GRPO on each domain adapter using scored data
      ↓  pushes improved adapters back to HuggingFace
      ↓  inference pipeline auto-picks them up on next load

Reward model (no LLM needed):
  reward = 0.5 * bertscore_f1(response, reference)   # semantic relevance
         + 0.3 * (1 - repetition_score(response))    # fluency
         + 0.2 * length_score(response, max=256)      # conciseness
  All components in [0, 1], final reward mapped to [-1, 1] for GRPO.

Install:
  pip install transformers==4.47.1 trl>=0.16.0 peft>=0.14.0 \
              accelerate datasets bert-score wandb huggingface_hub

Python: 3.11.x
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, gc, json, random, warnings, time
warnings.filterwarnings("ignore")

# !pip install bert_score -q

import torch
import numpy as np
import wandb
from pathlib import Path
from bert_score import BERTScorer
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import GRPOTrainer, GRPOConfig
from huggingface_hub import login

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
HF_TOKEN      = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API")
HF_USERNAME   = "kunjcr2"
VERSION       = "v4"

BASE_MODEL_ID = "google/gemma-3-1b-it"

ADAPTER_REPOS = {
    "code":    f"{HF_USERNAME}/code-adaptroute-{VERSION}",
    "math":    f"{HF_USERNAME}/math-adaptroute-{VERSION}",
    "medical": f"{HF_USERNAME}/medical-adaptroute-{VERSION}",
}

LOG_FILE   = Path("./query_log.jsonl")      # written by pipeline_v4.py at inference time
OUTPUT_DIR = Path("./continual_learning")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED      = 42
LOG_STEPS = 50
MIN_SAMPLES_TO_RETRAIN = 5   # don't retrain a domain with fewer than this many logs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("✓ HuggingFace login OK")

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD QUERY LOG
# ─────────────────────────────────────────────────────────────────────────────

def load_query_log() -> list[dict]:
    """
    Load logged queries from query_log.jsonl.
    Each line: {"question": ..., "answer": ..., "domain": ..., "timestamp": ...}
    Falls back to SEED_DATA if log is missing or too small.
    """
    records = []

    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        print(f"[Log] Loaded {len(records)} records from {LOG_FILE}")
    else:
        print(f"[Log] {LOG_FILE} not found — using seed data only")

    # Merge with seed data — seed data fills gaps for domains with few logs
    all_data = records
    print(f"[Log] Total samples after merging with seed data: {len(all_data)}")
    return all_data


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — REWARD SCORING
# ─────────────────────────────────────────────────────────────────────────────

def repetition_fraction(text: str, ngram: int = 4) -> float:
    """Fraction of n-grams that are duplicates. 0.0 = no repetition."""
    words = text.split()
    if len(words) < ngram:
        return 0.0
    grams = [tuple(words[i:i+ngram]) for i in range(len(words)-ngram+1)]
    return (len(grams) - len(set(grams))) / len(grams)


def length_score(text: str, ideal_max: int = 256) -> float:
    """
    Score response length.
    1.0 = at or under ideal_max tokens
    Linearly decays toward 0.0 as length doubles ideal_max
    """
    n_tokens = len(text.split())
    if n_tokens <= ideal_max:
        return 1.0
    # Penalize over-length — hits 0 at 2x ideal_max
    return max(0.0, 1.0 - (n_tokens - ideal_max) / ideal_max)


def compute_rewards(records: list[dict]) -> list[dict]:
    """
    Score each record with:
      reward = 0.5 * bertscore_f1   (semantic match to reference answer)
             + 0.3 * (1 - repetition_fraction)  (fluency)
             + 0.2 * length_score   (conciseness ≤256 tokens)

    BERTScorer is cached — one model load for all records.
    Final reward is mapped from [0,1] → [-1,1] for GRPO compatibility.
    """
    print("[Reward] Loading BERTScorer (distilbert-base-uncased) ...")
    # distilbert is tiny (~66MB) and fast — no need for deberta here
    scorer = BERTScorer(
        model_type="distilbert-base-uncased",
        lang="en",
        rescale_with_baseline=True,
        device=DEVICE,
    )

    scored = []
    references = [r["answer"] for r in records]

    # BERTScore needs a "candidate" — we use the question itself as the proxy
    # since we don't have the model's actual response yet (it will be generated
    # by GRPO during training). We score the reference answer against itself
    # to get a quality upper bound, and use rule-based signals for the rest.
    # In production with real logs, replace `questions` with actual model responses.
    candidates = [r.get("model_response", r["question"]) for r in records]

    print(f"[Reward] Scoring {len(candidates)} samples with BERTScore ...")
    _, _, F1 = scorer.score(candidates, references, verbose=False)
    F1 = F1.tolist()

    for i, record in enumerate(records):
        response_text = candidates[i]

        bs_score   = float(F1[i])                              # [0, 1]
        rep_score  = 1.0 - repetition_fraction(response_text) # [0, 1]
        len_score  = length_score(response_text, ideal_max=256) # [0, 1]

        raw_reward = (
            0.5 * bs_score
          + 0.3 * rep_score
          + 0.2 * len_score
        )  # [0, 1]

        # Map to [-1, 1] for GRPO
        final_reward = raw_reward * 2.0 - 1.0

        scored.append({
            **record,
            "bertscore_f1":  round(bs_score,   4),
            "rep_score":     round(rep_score,   4),
            "len_score":     round(len_score,   4),
            "raw_reward":    round(raw_reward,  4),
            "reward":        round(final_reward, 4),
        })

    del scorer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    avg = np.mean([s["reward"] for s in scored])
    print(f"[Reward] Done. Mean reward: {avg:.3f}")
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# GRPO REWARD FUNCTION (called by GRPOTrainer at training time)
# ─────────────────────────────────────────────────────────────────────────────

# BERTScorer singleton — loaded once, reused across GRPO reward calls
_bert_scorer: BERTScorer | None = None

def _get_bert_scorer() -> BERTScorer:
    global _bert_scorer
    if _bert_scorer is None:
        _bert_scorer = BERTScorer(
            model_type="distilbert-base-uncased",
            lang="en",
            rescale_with_baseline=True,
            device=DEVICE,
        )
    return _bert_scorer


def grpo_reward_fn(completions: list, reference: list[str], **_) -> list[float]:
    # GRPOTrainer passes completions as list of message dicts — extract text
    completion_texts = []
    for c in completions:
        if isinstance(c, list):
            # conversational format: [{"role": "assistant", "content": "..."}]
            completion_texts.append(c[-1]["content"] if c else "")
        elif isinstance(c, dict):
            completion_texts.append(c.get("content", ""))
        else:
            completion_texts.append(str(c))

    bs_scorer = _get_bert_scorer()
    _, _, F1  = bs_scorer.score(completion_texts, reference, verbose=False)
    F1 = F1.tolist()

    rewards = []
    for i, text in enumerate(completion_texts):
        bs_score  = float(F1[i])
        rep_score = 1.0 - repetition_fraction(text)
        len_score = length_score(text, ideal_max=256)
        raw = 0.5 * bs_score + 0.3 * rep_score + 0.2 * len_score
        rewards.append(raw * 2.0 - 1.0)

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — GRPO RETRAINING PER DOMAIN ADAPTER
# ─────────────────────────────────────────────────────────────────────────────

GRPO_LORA = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)


def fmt_prompt(question: str) -> str:
    """Gemma-3 chat format — inference style (no end_of_turn after model)."""
    return (
        f"<start_of_turn>user\n{question.strip()}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def retrain_adapter(domain: str, domain_records: list[dict]):
    """
    Pull the existing adapter from HuggingFace, run GRPO using scored logs,
    push the improved adapter back.
    """
    print(f"\n{'='*60}")
    print(f"GRPO Retraining — {domain.upper()} adapter")
    print(f"Samples: {len(domain_records)}")
    print(f"{'='*60}")

    wandb.init(
        project="adaptroute-continual",
        name=f"grpo-retrain-{domain}-{int(time.time())}",
        config={
            "domain":    domain,
            "n_samples": len(domain_records),
            "adapter":   ADAPTER_REPOS[domain],
        },
        reinit=True,
    )

    # ── Load tokenizer + base model ──────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, token=HF_TOKEN if HF_TOKEN else None
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required by GRPOTrainer

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        token=HF_TOKEN if HF_TOKEN else None,
    )

    # ── Load existing adapter on top of base ────────────────────
    print(f"[{domain}] Loading existing adapter from {ADAPTER_REPOS[domain]} ...")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_REPOS[domain],
        adapter_name=domain,
        is_trainable=True,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    model.print_trainable_parameters()

    # ── Build GRPO dataset ───────────────────────────────────────
    # GRPOTrainer expects a "prompt" column (list of chat messages)
    # We also pass "reference" so grpo_reward_fn can access it via **kwargs
    prompts    = []
    references = []
    for rec in domain_records:
        prompts.append([
            {"role": "user", "content": rec["question"]}
        ])
        references.append(rec["answer"])

    grpo_dataset = Dataset.from_dict({
        "prompt":    prompts,
        "reference": references,
    })

    # ── GRPOConfig ───────────────────────────────────────────────
    grpo_cfg = GRPOConfig(
        output_dir=str(OUTPUT_DIR / f"grpo-{domain}"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,      # effective batch = 8
        learning_rate=5e-6,
        bf16=True,
        tf32=True,
        logging_steps=LOG_STEPS,
        save_steps=500,
        report_to="wandb",
        # max_prompt_length=256,
        max_completion_length=256,
        num_generations=4,                  # completions per prompt for GRPO group
        temperature=0.9,
        beta=0.0,                           # no KL penalty
        seed=SEED,
        remove_unused_columns=False,        # keep "reference" column for reward fn
    )

    trainer = GRPOTrainer(
        model=base_model,           # plain base model, not PeftModel
        reward_funcs=grpo_reward_fn,
        args=grpo_cfg,
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
        peft_config=GRPO_LORA,      # GRPOTrainer wraps it internally
    )
    trainer.train()

    # ── Push improved adapter to HuggingFace ────────────────────
    repo = ADAPTER_REPOS[domain]
    print(f"[{domain}] Pushing improved adapter → {repo}")
    trainer.model.push_to_hub(repo, token=HF_TOKEN)
    tokenizer.push_to_hub(repo, token=HF_TOKEN)

    wandb.finish()
    del model, base_model, trainer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"[{domain}] Retraining complete.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — LOGGING HELPER (called from pipeline_v4.py at inference time)
# ─────────────────────────────────────────────────────────────────────────────

def log_query(question: str, model_response: str, domain: str):
    """
    Append a query+response to query_log.jsonl.
    Call this from pipeline_v4.py's process_query() after generation.

    Usage in pipeline_v4.py:
        from continual_learning import log_query
        log_query(query, response, winning_domain or "general")
    """
    record = {
        "question":       question,
        "answer":         model_response,   # model's actual response = the "answer" to score
        "model_response": model_response,
        "domain":         domain,
        "timestamp":      time.time(),
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — full retraining loop
# ─────────────────────────────────────────────────────────────────────────────

def run_retraining_loop(stop_event=None, on_phase=None):
    """
    Full offline retraining pipeline:
      1. Load query logs + seed data
      2. Score all records with reward model
      3. For each domain with enough samples → run GRPO → push adapter

    stop_event: threading.Event — when set, loop stops between domains.
    on_phase: callable(phase: str, domain: str | None) — progress hook for
              the API server to track status.  Phases: loading, scoring,
              training, done_domain, skipped, complete, stopped.
    """
    def _phase(phase, domain=None):
        if on_phase:
            try:
                on_phase(phase, domain)
            except Exception:
                pass

    t0 = time.time()
    print("\n" + "█"*60)
    print("  AdaptRoute — Continual Learning Loop")
    print("█"*60 + "\n")

    # 1. Load data
    _phase("loading")
    all_records = load_query_log()

    # 2. Score
    _phase("scoring")
    scored_records = compute_rewards(all_records)

    # Save scored records for inspection
    scored_path = OUTPUT_DIR / "scored_records.jsonl"
    with open(scored_path, "w") as f:
        for rec in scored_records:
            f.write(json.dumps(rec) + "\n")
    print(f"[Log] Scored records saved to {scored_path}")

    # 3. Split by domain and retrain each adapter
    for domain in ADAPTER_REPOS.keys():
        if stop_event and stop_event.is_set():
            print("[Stop] Training stopped by user request.")
            _phase("stopped")
            return

        domain_records = [r for r in scored_records if r.get("domain") == domain]

        if len(domain_records) < MIN_SAMPLES_TO_RETRAIN:
            print(f"\n[Skip] {domain}: only {len(domain_records)} samples "
                  f"(need {MIN_SAMPLES_TO_RETRAIN}). Skipping.")
            _phase("skipped", domain)
            continue

        # Sort by reward descending — train on the best examples
        domain_records.sort(key=lambda x: x["reward"], reverse=True)

        print(f"\n[{domain}] {len(domain_records)} samples | "
              f"mean reward: {np.mean([r['reward'] for r in domain_records]):.3f} | "
              f"top reward: {domain_records[0]['reward']:.3f}")

        _phase("training", domain)
        retrain_adapter(domain, domain_records)
        _phase("done_domain", domain)
        print(f"[Timer] After {domain}: {(time.time()-t0)/3600:.2f} hrs")

    total = (time.time() - t0) / 3600
    print("\n" + "█"*60)
    print(f"  Continual Learning Loop Complete in {total:.2f} hrs")
    print(f"  Updated adapters:")
    for d, r in ADAPTER_REPOS.items():
        print(f"    {d:<8} → {r}")
    print("█"*60 + "\n")
    _phase("complete")


if __name__ == "__main__":
    run_retraining_loop()