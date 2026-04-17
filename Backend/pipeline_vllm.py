# !pip install vllm -q

import os
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download, login
import time

# ==============================================================================
# pipeline_vllm.py
# Drop-in replacement for pipeline.py using vLLM for generation.
# Firewall + Gating stay on HF transformers (they're tiny, already fast).
# Only the base model generation switches to vLLM for 10x speed gains.
#
# Usage:
#   pip install vllm
#   python app.py  (with: import pipeline_vllm as pipeline)
# ==============================================================================

# Force vLLM to use the stable v0 engine.
# The v1 engine (default in vLLM 0.7+) has known LoRA bugs with per-request switching.
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ==============================================================================
# HUGGING FACE AUTH — reads HF_TOKEN from environment variable
# Set it on the SSH server:  export HF_TOKEN="hf_your_token_here"
# ==============================================================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("✓ HuggingFace login OK")
else:
    print("⚠ HF_TOKEN not set — public models only")

# ==============================================================================
# CONFIGURATION — keep in sync with pipeline.py
# ==============================================================================
FIREWALL_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
GATING_MODEL   = "kunjcr2/gating-bert-adaptroute"
BASE_MODEL     = "Qwen/Qwen2.5-1.5B"

ADAPTER_REPOS = {
    "code":    "kunjcr2/code-adaptroute-v3",
    "math":    "kunjcr2/math-adaptroute-v3",
    "qa":      "kunjcr2/qa-adaptroute-v3",
    "medical": "kunjcr2/medical-adaptroute-v3",
}

# vLLM LoRARequest needs a unique integer ID per adapter name
ADAPTER_IDS = {
    "code":    1,
    "math":    2,
    "qa":      3,
    "medical": 4,
}

ADAPTERS_DIR = os.path.abspath(os.path.join(os.getcwd(), "Adapters"))
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==============================================================================
# Global state
# ==============================================================================
global_systems = {
    "firewall_model":     None,
    "firewall_tokenizer": None,
    "gating_model":       None,
    "gating_tokenizer":   None,
    "vllm_engine":        None,   # replaces base_model + base_tokenizer
}

SAMPLING_PARAMS = SamplingParams(
    max_tokens=256,
    temperature=0.0,          # greedy — same as do_sample=False
    repetition_penalty=1.5,
    stop=["<|im_end|>"],
)

# ==============================================================================
# prepare() — force re-download adapters from HF for a clean state
# ==============================================================================
def prepare():
    """
    Force re-downloads all adapters from HuggingFace into ADAPTERS_DIR.
    Deletes the existing Adapters folder first to ensure clean state.
    """
    # Always wipe and re-download for a clean state
    if os.path.exists(ADAPTERS_DIR):
        print(f"Removing existing Adapters folder: {ADAPTERS_DIR}")
        shutil.rmtree(ADAPTERS_DIR)
        print("✓ Deleted")

    os.makedirs(ADAPTERS_DIR, exist_ok=True)
    print(f"Downloading all adapters to {ADAPTERS_DIR}...")

    for domain, repo_id in ADAPTER_REPOS.items():
        local_path = os.path.join(ADAPTERS_DIR, domain)
        print(f"  Downloading {repo_id} → {local_path} ...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_path,
            ignore_patterns=["*.msgpack", "*.h5"],
            token=HF_TOKEN if HF_TOKEN else None,
        )
        print(f"  ✓ {domain} done")

    print("✓ All adapters downloaded")

# ==============================================================================
# load_all_models() — Firewall + Gating on HF, base model on vLLM
# ==============================================================================
def load_all_models():
    global global_systems

    # 1. Firewall (ProtectAI DeBERTa — ~180MB, runs on CPU to keep GPU free for vLLM)
    print("Loading Firewall on CPU...")
    global_systems["firewall_tokenizer"] = AutoTokenizer.from_pretrained(FIREWALL_MODEL)
    global_systems["firewall_model"] = AutoModelForSequenceClassification.from_pretrained(FIREWALL_MODEL).to("cpu")
    global_systems["firewall_model"].eval()

    # 2. Gating Network (DistilBERT — ~66MB, runs on CPU to keep GPU free for vLLM)
    print("Loading Gating Network on CPU...")
    global_systems["gating_tokenizer"] = AutoTokenizer.from_pretrained(GATING_MODEL)
    global_systems["gating_model"] = AutoModelForSequenceClassification.from_pretrained(GATING_MODEL).to("cpu")
    global_systems["gating_model"].eval()

    # 3. Base model on vLLM — give it the full GPU, no HF models sharing VRAM
    print("Loading Base Model into vLLM engine (this takes ~30s)...")
    global_systems["vllm_engine"] = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=64,         # v4 adapters use r=32 and r=64
        max_loras=4,              # pre-allocate slots for all 4 domain adapters
        max_cpu_loras=4,          # keep all adapters cached in CPU RAM between requests
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        enforce_eager=True,       # skip CUDA graph compilation — fixes EngineDeadError with LoRA
        max_num_seqs=1,           # single-user inference, no need for batching overhead
    )

    print("All models loaded. Ready.")

# ==============================================================================
# process_query() — Firewall → Gating → vLLM generation
# ==============================================================================
def process_query(query: str) -> dict:
    if any(v is None for v in global_systems.values()):
        return {"status": "error", "message": "Models not loaded. Call load_all_models() first."}

    t_start = time.time()

    # ── 1. Firewall ────────────────────────────────────────────────────────────
    fw_tok   = global_systems["firewall_tokenizer"]
    fw_model = global_systems["firewall_model"]

    fw_inputs = fw_tok(query, return_tensors="pt", truncation=True, max_length=512).to(fw_model.device)
    fw_inputs.pop("token_type_ids", None)  # DistilBERT/DeBERTa don't use this
    with torch.no_grad():
        fw_logits = fw_model(**fw_inputs).logits

    fw_label = fw_model.config.id2label.get(fw_logits.argmax(dim=-1).item(), "SAFE")

    if fw_label == "INJECTION":
        return {
            "status":         "blocked",
            "message":        "Your query was flagged as a potential prompt injection attempt and could not be processed. Please rephrase your request.",
            "firewall_label": fw_label,
        }

    t_fw = time.time()

    # ── 2. Gating Network ──────────────────────────────────────────────────────
    gate_tok   = global_systems["gating_tokenizer"]
    gate_model = global_systems["gating_model"]

    gate_inputs = gate_tok(query, return_tensors="pt", truncation=True, max_length=512).to(gate_model.device)
    gate_inputs.pop("token_type_ids", None)  # DistilBERT doesn't use this
    with torch.no_grad():
        gate_logits = gate_model(**gate_inputs).logits

    probs       = torch.softmax(gate_logits, dim=-1).squeeze()
    best_idx    = probs.argmax().item()
    best_prob   = probs[best_idx].item()
    id2label    = gate_model.config.id2label

    # THRESHOLD CHECK: If confidence is below 0.9, use base model without adapter
    if best_prob < 0.9:
        print(f"Gating confidence {best_prob:.4f} below threshold (0.9). Using base model without adapter.")
        winning_domain = None
        winning_label = "base_model"
    else:
        winning_label = id2label.get(best_idx, "").lower()
        winning_domain = None

        for domain in ADAPTER_REPOS:
            if domain in winning_label:
                winning_domain = domain
                break

        print(f"Using adapter: {winning_domain}")

        if not winning_domain:
            return {"status": "error", "message": f"Gating network returned unknown label: {winning_label}"}

    t_gate = time.time()

    # ── 3. vLLM Generation (with or without LoRA) ─────────────────────────────
    formatted_prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

    # Dynamic max_tokens based on domain
    max_tokens_map = {"medical": 256, "code": 256, "math": 128, "qa": 64}
    max_new_tokens = max_tokens_map.get(winning_domain, 128) if winning_domain else 128

    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        repetition_penalty=1.5,
        stop=["<|im_end|>"],
    )

    if winning_domain is not None:
        # Generate with LoRA adapter
        local_adapter_path = os.path.join(ADAPTERS_DIR, winning_domain)
        adapter_source = local_adapter_path if os.path.exists(local_adapter_path) else ADAPTER_REPOS[winning_domain]

        lora_request = LoRARequest(
            lora_name   = winning_domain,
            lora_int_id = ADAPTER_IDS[winning_domain],
            lora_path   = adapter_source,
        )
        outputs = global_systems["vllm_engine"].generate(
            formatted_prompt,
            sampling,
            lora_request=lora_request,
        )
    else:
        # Generate with base model only (no adapter)
        outputs = global_systems["vllm_engine"].generate(
            formatted_prompt,
            sampling,
        )

    response = outputs[0].outputs[0].text.strip()
    generated_tokens = len(outputs[0].outputs[0].token_ids)

    t_gen = time.time()
    t_total = t_gen - t_start

    print("\n--- vLLM INFERENCE PROFILING ---")
    print(f"Firewall:   {t_fw - t_start:.2f}s")
    print(f"Gating:     {t_gate - t_fw:.2f}s")
    print(f"Generation: {t_gen - t_gate:.2f}s  ({generated_tokens} tokens, {generated_tokens / (t_gen - t_gate + 1e-6):.1f} tok/s)")
    print(f"Total:      {t_total:.2f}s")
    print("--------------------------------\n")

    gating_scores = {
        id2label.get(i, str(i)).lower(): round(probs[i].item(), 4)
        for i in range(len(probs))
    }

    return {
        "status":            "success",
        "response":          response,
        "adapter_used":      winning_domain if winning_domain else "base_model",
        "gating_confidence": round(best_prob, 4),
        "gating_scores":     gating_scores,
        "firewall_label":    fw_label,
        "time_seconds":      round(t_total, 2),
    }

# When run directly, execute a quick test
if __name__ == "__main__":
    prepare()
    load_all_models()
    result = process_query("Write a function to convert a list into a dictionary with index as keys and values as values")
    print(result)