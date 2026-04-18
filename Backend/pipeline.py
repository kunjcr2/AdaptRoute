import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel
import os
from huggingface_hub import snapshot_download, login
import time
from collections import OrderedDict
from google.colab import userdata

# ==============================================================================
# HUGGING FACE AUTH
# ==============================================================================
HF_TOKEN = userdata.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("✓ HuggingFace login OK")
else:
    print("⚠ HF_TOKEN not set — public models only")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
FIREWALL_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
GATING_MODEL   = "kunjcr2/gating-bert-adaptroute-v4"
BASE_MODEL     = "google/gemma-3-1b-it"

ADAPTER_REPOS = {
    "code":    "kunjcr2/code-adaptroute-v4",
    "math":    "kunjcr2/math-adaptroute-v4",
    "medical": "kunjcr2/medical-adaptroute-v4",
}

ADAPTERS_DIR = os.path.abspath(os.path.join(os.getcwd(), "Adapters"))
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==============================================================================
# ROUTING THRESHOLDS
# >0.85       → hard route to single adapter (fast path)
# 0.60–0.85   → soft blend top-2 adapters weighted by their confidence scores
# <0.60       → base model only, no adapter
# ==============================================================================
HARD_ROUTE_THRESHOLD = 0.85
BLEND_THRESHOLD      = 0.60

# Cache loaded adapter weight deltas so blending doesn't re-download every call
# Structure: { domain_name: { param_name: lora_delta_tensor } }
_adapter_delta_cache: dict = {}

# Global dictionary to keep models loaded in RAM
global_systems = {
    "firewall_model":      None,
    "firewall_tokenizer":  None,
    "gating_model":        None,
    "gating_tokenizer":    None,
    "base_model":          None,
    "base_tokenizer":      None,
}


# ==============================================================================
# PREPARE — download adapters
# ==============================================================================
def prepare():
    """
    Force re-downloads all adapters from HuggingFace into ADAPTERS_DIR.
    Deletes the existing Adapters folder first to ensure clean state.
    """
    import shutil

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
# MODEL LOADING
# ==============================================================================
def load_all_models():
    """
    Loads the Firewall, Gating Network, and Base Model into RAM.
    Adapters are loaded on first use and cached.
    """
    global global_systems

    # 1. Firewall
    print("Loading Firewall Model...")
    global_systems["firewall_tokenizer"] = AutoTokenizer.from_pretrained(FIREWALL_MODEL)
    global_systems["firewall_model"]     = AutoModelForSequenceClassification.from_pretrained(
        FIREWALL_MODEL
    ).to(DEVICE)
    global_systems["firewall_model"].eval()

    # 2. Gating Network
    print("Loading Gating Network...")
    global_systems["gating_tokenizer"] = AutoTokenizer.from_pretrained(GATING_MODEL)
    global_systems["gating_model"]     = AutoModelForSequenceClassification.from_pretrained(
        GATING_MODEL
    ).to(DEVICE)
    global_systems["gating_model"].eval()

    # 3. Base Model — Gemma 3 1B (text-only → AutoModelForCausalLM)
    print("Loading Base Model (Gemma 3 1B, bfloat16, SDPA)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        token=HF_TOKEN if HF_TOKEN else None,
    ).to(DEVICE)
    base_model.config.use_cache = True

    base_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    base_tokenizer.padding_side = "right"

    global_systems["base_model"]     = base_model
    global_systems["base_tokenizer"] = base_tokenizer
    global_systems["base_model"].eval()

    print("All models loaded. Adapters will load on first use and be cached.")


# ==============================================================================
# SOFT BLENDING HELPERS
# ==============================================================================

def _get_adapter_source(domain: str) -> str:
    """Return local path if downloaded, else HF repo id."""
    local = os.path.join(ADAPTERS_DIR, domain)
    return local if os.path.exists(local) else ADAPTER_REPOS[domain]


def _load_adapter_deltas(domain: str, base_model: torch.nn.Module) -> dict:
    """
    Load a LoRA adapter and extract its weight deltas (B @ A * scaling).
    Caches results so each adapter is only computed once per session.

    Delta = what the adapter *adds* on top of the base weights.
    For LoRA: delta[layer] = lora_B @ lora_A * (alpha / r)
    """
    if domain in _adapter_delta_cache:
        return _adapter_delta_cache[domain]

    print(f"[Blend] Loading adapter deltas for '{domain}' ...")
    source = _get_adapter_source(domain)

    # Load adapter temporarily to extract deltas
    peft_model = PeftModel.from_pretrained(
        base_model, source, adapter_name=domain
    )

    deltas = {}
    adapter_cfg = peft_model.peft_config[domain]
    scaling     = adapter_cfg.lora_alpha / adapter_cfg.r

    for name, module in peft_model.named_modules():
        # lora_A and lora_B exist on LoRA-wrapped layers
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_A = module.lora_A[domain].weight  # (r, in)
            lora_B = module.lora_B[domain].weight  # (out, r)
            delta  = (lora_B @ lora_A) * scaling   # (out, in)
            deltas[name] = delta.detach().clone()

    # Unload — we only needed the weights
    del peft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _adapter_delta_cache[domain] = deltas
    print(f"[Blend] Cached {len(deltas)} delta tensors for '{domain}'")
    return deltas


def _apply_blended_delta(base_model: torch.nn.Module, blended_deltas: dict):
    """Temporarily add blended deltas to base model weights in-place."""
    for name, module in base_model.named_modules():
        if name in blended_deltas:
            # Find the weight parameter (usually .weight on Linear layers)
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data += blended_deltas[name].to(module.weight.dtype)


def _remove_blended_delta(base_model: torch.nn.Module, blended_deltas: dict):
    """Remove the blended deltas after generation (restore base weights)."""
    for name, module in base_model.named_modules():
        if name in blended_deltas:
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data -= blended_deltas[name].to(module.weight.dtype)


def _build_blended_deltas(
    domain_a: str, weight_a: float,
    domain_b: str, weight_b: float,
    base_model: torch.nn.Module,
) -> dict:
    """
    Interpolate two adapter deltas by their gating confidence weights.
    blended_delta = weight_a * delta_a + weight_b * delta_b
    """
    deltas_a = _load_adapter_deltas(domain_a, base_model)
    deltas_b = _load_adapter_deltas(domain_b, base_model)

    blended = {}
    all_keys = set(deltas_a.keys()) | set(deltas_b.keys())
    for key in all_keys:
        d_a = deltas_a.get(key, None)
        d_b = deltas_b.get(key, None)
        if d_a is not None and d_b is not None:
            blended[key] = weight_a * d_a + weight_b * d_b
        elif d_a is not None:
            blended[key] = weight_a * d_a
        elif d_b is not None:
            blended[key] = weight_b * d_b

    print(f"[Blend] {domain_a}({weight_a:.2f}) + {domain_b}({weight_b:.2f}) → {len(blended)} blended layers")
    return blended


# ==============================================================================
# MAIN INFERENCE
# ==============================================================================
def process_query(query: str) -> dict:
    """
    Passes a single query through the firewall, gating network, routes to
    the appropriate adapter (hard route / soft blend / base only), generates
    a response, and returns a structured dict identical to the old pipeline.

    Routing logic:
      confidence > 0.85  → hard route to top-1 adapter
      0.60–0.85          → soft blend top-2 adapters by confidence weight
      < 0.60             → base model only
    """
    if any(m is None for m in global_systems.values()):
        return {"status": "error", "message": "Models not loaded. Call load_all_models() first."}

    t_start = time.time()

    # ──────────────────────────────────────────────────────────────
    # 1. Firewall Check
    # ──────────────────────────────────────────────────────────────
    fw_tokenizer = global_systems["firewall_tokenizer"]
    fw_model     = global_systems["firewall_model"]

    fw_inputs = fw_tokenizer(
        query, return_tensors="pt", truncation=True, max_length=512
    ).to(fw_model.device)

    with torch.no_grad():
        fw_outputs = fw_model(**fw_inputs)

    predicted_fw_class_id = fw_outputs.logits.argmax(dim=-1).item()
    fw_label = fw_model.config.id2label.get(predicted_fw_class_id, "SAFE")

    if fw_label == "INJECTION":
        return {
            "status":         "blocked",
            "message":        "Your query was flagged as a potential prompt injection attempt and could not be processed. Please rephrase your request.",
            "firewall_label": fw_label,
        }

    # ──────────────────────────────────────────────────────────────
    # 2. Gating Network
    # ──────────────────────────────────────────────────────────────
    gating_tokenizer = global_systems["gating_tokenizer"]
    gating_model     = global_systems["gating_model"]

    gate_inputs = gating_tokenizer(
        query, return_tensors="pt", truncation=True, max_length=512
    ).to(gating_model.device)

    if "token_type_ids" in gate_inputs:
        del gate_inputs["token_type_ids"]

    with torch.no_grad():
        gate_outputs = gating_model(**gate_inputs)

    probs         = torch.softmax(gate_outputs.logits, dim=-1).squeeze()
    gate_id2label = gating_model.config.id2label

    # Sort all domains by confidence descending
    sorted_idx  = probs.argsort(descending=True).tolist()
    top1_idx    = sorted_idx[0]
    top2_idx    = sorted_idx[1]
    top1_prob   = probs[top1_idx].item()
    top2_prob   = probs[top2_idx].item()
    top1_label  = gate_id2label.get(top1_idx, "").lower()
    top2_label  = gate_id2label.get(top2_idx, "").lower()

    def label_to_domain(label: str):
        for domain in ADAPTER_REPOS.keys():
            if domain.lower() in label:
                return domain
        return None

    top1_domain = label_to_domain(top1_label)
    top2_domain = label_to_domain(top2_label)

    # ── Decide routing mode ────────────────────────────────────────
    base_model     = global_systems["base_model"]
    base_tokenizer = global_systems["base_tokenizer"]

    routing_mode   = None   # "hard" | "blend" | "base"
    winning_domain = None
    blended_deltas = None

    if top1_prob >= HARD_ROUTE_THRESHOLD and top1_domain:
        # ── Hard route: single adapter ─────────────────────────────
        routing_mode   = "hard"
        winning_domain = top1_domain
        print(f"[Route] HARD → {winning_domain} (conf={top1_prob:.3f})")

    elif top1_prob >= BLEND_THRESHOLD and top1_domain and top2_domain:
        # ── Soft blend: interpolate top-2 adapter deltas ───────────
        routing_mode = "blend"
        # Re-normalise the top-2 probabilities so they sum to 1
        total      = top1_prob + top2_prob
        weight_a   = top1_prob / total
        weight_b   = top2_prob / total
        winning_domain = top1_domain   # primary domain for metadata

        print(f"[Route] BLEND → {top1_domain}({weight_a:.2f}) + {top2_domain}({weight_b:.2f})")

        # Get a clean base model (not a PeftModel) for delta application
        raw_base = base_model.base_model if isinstance(base_model, PeftModel) else base_model

        blended_deltas = _build_blended_deltas(
            top1_domain, weight_a,
            top2_domain, weight_b,
            raw_base,
        )

    else:
        # ── Base model only ────────────────────────────────────────
        routing_mode   = "base"
        winning_domain = None
        print(f"[Route] BASE only (top conf={top1_prob:.3f} below {BLEND_THRESHOLD})")

    # ──────────────────────────────────────────────────────────────
    # 3. Adapter Loading (hard route only)
    # ──────────────────────────────────────────────────────────────
    if routing_mode == "hard":
        source = _get_adapter_source(winning_domain)

        if not isinstance(base_model, PeftModel):
            base_model = PeftModel.from_pretrained(
                base_model, source, adapter_name=winning_domain
            )
            global_systems["base_model"] = base_model
        else:
            if winning_domain not in base_model.peft_config:
                base_model.load_adapter(source, adapter_name=winning_domain)
            base_model.set_adapter(winning_domain)

    # ──────────────────────────────────────────────────────────────
    # 4. Generation
    # ──────────────────────────────────────────────────────────────
    max_tokens_map = {"medical": 256, "code": 256, "math": 128}
    max_new_tokens = max_tokens_map.get(winning_domain, 128) if winning_domain else 128

    # Gemma-3 prompt format
    formatted_prompt = (
        f"<start_of_turn>user\n{query}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    enc = base_tokenizer(formatted_prompt, return_tensors="pt").to(base_model.device)

    # Stop tokens
    stop_tokens     = [base_tokenizer.eos_token_id]
    end_of_turn_id  = base_tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end_of_turn_id and end_of_turn_id != base_tokenizer.unk_token_id:
        stop_tokens.append(end_of_turn_id)

    ngram_size = 5 if winning_domain == "math" else 3

    # Apply blended deltas before generation, remove after
    if routing_mode == "hard":
        raw_base = base_model.base_model if isinstance(base_model, PeftModel) else base_model
        base_model.set_adapter(winning_domain)
        base_model.merge_adapter()

    elif routing_mode == "blend":
        raw_base = base_model.base_model if isinstance(base_model, PeftModel) else base_model
        _apply_blended_delta(raw_base, blended_deltas)

    with torch.inference_mode():
        out = base_model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            no_repeat_ngram_size=ngram_size,
            pad_token_id=base_tokenizer.pad_token_id,
            eos_token_id=stop_tokens,
        )

    # Restore weights
    if routing_mode == "hard":
        base_model.unmerge_adapter()
    elif routing_mode == "blend":
        _remove_blended_delta(raw_base, blended_deltas)

    # ──────────────────────────────────────────────────────────────
    # 5. Decode + Post-process (identical to old pipeline)
    # ──────────────────────────────────────────────────────────────
    response = base_tokenizer.decode(
        out[0][enc["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    if winning_domain == "code":
        parts = response.split("#")
        if len(parts) > 1:
            response = "#".join(parts[:-1]).strip()
        response = response.rstrip()
    else:
        parts = response.split(".")
        if len(parts) > 1:
            response = ".".join(parts[:-1]).strip()
        parts = response.split("\n")
        if len(parts) > 1:
            response = "\n".join(parts[:-1]).strip()
        response = response.rstrip()

    t_total = time.time() - t_start

    gating_scores = {
        gate_id2label.get(i, str(i)).lower(): round(probs[i].item(), 4)
        for i in range(len(probs))
    }

    return {
        "status":             "success",
        "response":           response,
        "adapter_used":       winning_domain if winning_domain else "base_model",
        "routing_mode":       routing_mode,           # new field: "hard" | "blend" | "base"
        "gating_confidence":  round(top1_prob, 4),
        "gating_scores":      gating_scores,
        "firewall_label":     fw_label,
        "time_seconds":       round(t_total, 2),
    }

prepare()
load_all_models()