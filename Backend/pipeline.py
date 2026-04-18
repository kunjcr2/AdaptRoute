import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, TextIteratorStreamer
from peft import PeftModel
import os
from huggingface_hub import snapshot_download, login
import time
import threading
try:
    from google.colab import userdata
except ImportError:
    userdata = None

# ==============================================================================
# HUGGING FACE AUTH
# ==============================================================================
HF_TOKEN = None
if userdata:
    try:
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        pass

if not HF_TOKEN:
    HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("[Auth] HF_TOKEN login OK")
else:
    print("[Auth] HF_TOKEN not set - public models only or use 'huggingface-cli login'")

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
# ==============================================================================
HARD_ROUTE_THRESHOLD = 0.85
BLEND_THRESHOLD      = 0.60

_adapter_delta_cache: dict = {}

# Serializes generation so adapter weight mutations don't collide across requests.
_model_lock = threading.Lock()

global_systems = {
    "firewall_model":      None,
    "firewall_tokenizer":  None,
    "gating_model":        None,
    "gating_tokenizer":    None,
    "base_model":          None,
    "base_tokenizer":      None,
}


# ==============================================================================
# PREPARE - download adapters
# ==============================================================================
def prepare():
    import shutil

    if os.path.exists(ADAPTERS_DIR):
        print(f"Removing existing Adapters folder: {ADAPTERS_DIR}")
        shutil.rmtree(ADAPTERS_DIR)
        print("Deleted")

    os.makedirs(ADAPTERS_DIR, exist_ok=True)
    print(f"Downloading all adapters to {ADAPTERS_DIR}...")

    for domain, repo_id in ADAPTER_REPOS.items():
        local_path = os.path.join(ADAPTERS_DIR, domain)
        print(f"  Downloading {repo_id} -> {local_path} ...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_path,
            ignore_patterns=["*.msgpack", "*.h5"],
            token=HF_TOKEN if HF_TOKEN else None,
        )
        print(f"  {domain} done")

    print("All adapters downloaded")


# ==============================================================================
# MODEL LOADING
# ==============================================================================
def load_all_models():
    global global_systems

    print("Loading Firewall Model...")
    global_systems["firewall_tokenizer"] = AutoTokenizer.from_pretrained(FIREWALL_MODEL)
    global_systems["firewall_model"]     = AutoModelForSequenceClassification.from_pretrained(
        FIREWALL_MODEL
    ).to(DEVICE)
    global_systems["firewall_model"].eval()

    print("Loading Gating Network...")
    global_systems["gating_tokenizer"] = AutoTokenizer.from_pretrained(GATING_MODEL)
    global_systems["gating_model"]     = AutoModelForSequenceClassification.from_pretrained(
        GATING_MODEL
    ).to(DEVICE)
    global_systems["gating_model"].eval()

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
    local = os.path.join(ADAPTERS_DIR, domain)
    return local if os.path.exists(local) else ADAPTER_REPOS[domain]


def _load_adapter_deltas(domain: str, base_model: torch.nn.Module) -> dict:
    if domain in _adapter_delta_cache:
        return _adapter_delta_cache[domain]

    print(f"[Blend] Loading adapter deltas for '{domain}' ...")
    source = _get_adapter_source(domain)

    peft_model = PeftModel.from_pretrained(
        base_model, source, adapter_name=domain
    )

    deltas = {}
    adapter_cfg = peft_model.peft_config[domain]
    scaling     = adapter_cfg.lora_alpha / adapter_cfg.r

    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_A = module.lora_A[domain].weight
            lora_B = module.lora_B[domain].weight
            delta  = (lora_B @ lora_A) * scaling
            deltas[name] = delta.detach().clone()

    del peft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _adapter_delta_cache[domain] = deltas
    print(f"[Blend] Cached {len(deltas)} delta tensors for '{domain}'")
    return deltas


def _apply_blended_delta(base_model: torch.nn.Module, blended_deltas: dict):
    for name, module in base_model.named_modules():
        if name in blended_deltas:
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data += blended_deltas[name].to(module.weight.dtype)


def _remove_blended_delta(base_model: torch.nn.Module, blended_deltas: dict):
    for name, module in base_model.named_modules():
        if name in blended_deltas:
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data -= blended_deltas[name].to(module.weight.dtype)


def _build_blended_deltas(
    domain_a: str, weight_a: float,
    domain_b: str, weight_b: float,
    base_model: torch.nn.Module,
) -> dict:
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

    print(f"[Blend] {domain_a}({weight_a:.2f}) + {domain_b}({weight_b:.2f}) -> {len(blended)} blended layers")
    return blended


# ==============================================================================
# SHARED ROUTING LOGIC
# ==============================================================================
def _run_firewall_and_gating(query: str):
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
        return fw_label, None

    gating_tokenizer = global_systems["gating_tokenizer"]
    gating_model     = global_systems["gating_model"]

    gate_inputs = gating_tokenizer(
        query, return_tensors="pt", truncation=True, max_length=512
    ).to(gating_model.device)

    if "token_type_ids" in gate_inputs:
        del gate_inputs["token_type_ids"]

    with torch.no_grad():
        gate_outputs = gating_model(**gate_inputs)

    GATE_TEMPERATURE = 3.0
    probs = torch.softmax(gate_outputs.logits / GATE_TEMPERATURE, dim=-1).squeeze()

    gate_id2label = gating_model.config.id2label

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

    if top1_prob >= HARD_ROUTE_THRESHOLD and top1_domain:
        routing_mode   = "hard"
        winning_domain = top1_domain
    elif top1_prob >= BLEND_THRESHOLD and top1_domain and top2_domain:
        routing_mode   = "blend"
        winning_domain = top1_domain
    else:
        routing_mode   = "base"
        winning_domain = None

    gating_scores = {
        gate_id2label.get(i, str(i)).lower(): round(probs[i].item(), 6)
        for i in range(len(probs))
    }

    return fw_label, {
        "routing_mode":   routing_mode,
        "winning_domain": winning_domain,
        "top1_prob":      top1_prob,
        "top2_prob":      top2_prob,
        "top1_domain":    top1_domain,
        "top2_domain":    top2_domain,
        "gating_scores":  gating_scores,
    }


def _setup_adapter_for_generation(routing_info: dict):
    """
    Mutates the base model in place to apply the routed adapter.
    Returns (model, cleanup_fn). cleanup_fn MUST be called to restore base weights.
    """
    base_model     = global_systems["base_model"]
    routing_mode   = routing_info["routing_mode"]
    winning_domain = routing_info["winning_domain"]

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

        base_model.merge_adapter()

        def cleanup():
            base_model.unmerge_adapter()

        return base_model, cleanup

    elif routing_mode == "blend":
        raw_base = base_model.base_model if isinstance(base_model, PeftModel) else base_model
        blended_deltas = _build_blended_deltas(
            routing_info["top1_domain"],
            routing_info["top1_prob"] / (routing_info["top1_prob"] + routing_info["top2_prob"]),
            routing_info["top2_domain"],
            routing_info["top2_prob"] / (routing_info["top1_prob"] + routing_info["top2_prob"]),
            raw_base,
        )
        _apply_blended_delta(raw_base, blended_deltas)

        def cleanup():
            _remove_blended_delta(raw_base, blended_deltas)

        return base_model, cleanup

    else:  # base
        def cleanup():
            pass
        return base_model, cleanup


def _postprocess_response(response: str, winning_domain: str | None) -> str:
    return response.replace("<end_of_turn>", "").rstrip()


# ==============================================================================
# NON-STREAMING INFERENCE
# ==============================================================================
def process_query(query: str) -> dict:
    if any(m is None for m in global_systems.values()):
        return {"status": "error", "message": "Models not loaded. Call load_all_models() first."}

    t_start = time.time()

    fw_label, routing_info = _run_firewall_and_gating(query)

    if fw_label == "INJECTION":
        return {
            "status":         "blocked",
            "message":        "Your query was flagged as a potential prompt injection attempt and could not be processed. Please rephrase your request.",
            "firewall_label": fw_label,
        }

    base_tokenizer = global_systems["base_tokenizer"]
    winning_domain = routing_info["winning_domain"]
    routing_mode   = routing_info["routing_mode"]

    print(f"[Route] {routing_mode.upper()} -> {winning_domain or 'base'} (conf={routing_info['top1_prob']:.3f})")

    with _model_lock:
        model, cleanup = _setup_adapter_for_generation(routing_info)

        try:
            max_new_tokens = 2048

            formatted_prompt = (
                f"<start_of_turn>user\n{query}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            enc = base_tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            stop_tokens    = [base_tokenizer.eos_token_id]
            end_of_turn_id = base_tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if end_of_turn_id and end_of_turn_id != base_tokenizer.unk_token_id:
                stop_tokens.append(end_of_turn_id)

            ngram_size = 5 if winning_domain == "math" else 3

            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    no_repeat_ngram_size=ngram_size,
                    pad_token_id=base_tokenizer.pad_token_id,
                    eos_token_id=stop_tokens,
                )

            response = base_tokenizer.decode(
                out[0][enc["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
        finally:
            cleanup()

    response = _postprocess_response(response, winning_domain)
    t_total  = time.time() - t_start

    return {
        "status":             "success",
        "response":           response,
        "adapter_used":       winning_domain if winning_domain else "base_model",
        "routing_mode":       routing_mode,
        "gating_confidence":  round(routing_info["top1_prob"], 6),
        "gating_scores":      routing_info["gating_scores"],
        "firewall_label":     fw_label,
        "time_seconds":       round(t_total, 2),
    }


# ==============================================================================
# STREAMING INFERENCE
# ==============================================================================
def process_query_stream(query: str):
    """
    Generator that yields dicts. Each dict has a 'type' field:
        {"type": "meta", ...routing info}
        {"type": "token", "text": "..."}
        {"type": "done",  "time_seconds": float, "full_response": "..."}
        {"type": "blocked", "message": str, "firewall_label": "INJECTION"}
        {"type": "error",   "message": str}
    """
    if any(m is None for m in global_systems.values()):
        yield {"type": "error", "message": "Models not loaded. Call load_all_models() first."}
        return

    t_start = time.time()

    try:
        fw_label, routing_info = _run_firewall_and_gating(query)
    except Exception as e:
        yield {"type": "error", "message": f"Firewall/gating failed: {e}"}
        return

    if fw_label == "INJECTION":
        yield {
            "type":           "blocked",
            "message":        "Your query was flagged as a potential prompt injection attempt and could not be processed. Please rephrase your request.",
            "firewall_label": fw_label,
        }
        return

    base_tokenizer = global_systems["base_tokenizer"]
    winning_domain = routing_info["winning_domain"]
    routing_mode   = routing_info["routing_mode"]

    print(f"[Stream] {routing_mode.upper()} -> {winning_domain or 'base'} (conf={routing_info['top1_prob']:.3f})")

    yield {
        "type":              "meta",
        "adapter_used":      winning_domain if winning_domain else "base_model",
        "routing_mode":      routing_mode,
        "gating_confidence": round(routing_info["top1_prob"], 6),
        "gating_scores":     routing_info["gating_scores"],
        "firewall_label":    fw_label,
    }

    with _model_lock:
        model, cleanup = _setup_adapter_for_generation(routing_info)

        try:
            max_new_tokens = 2048

            formatted_prompt = (
                f"<start_of_turn>user\n{query}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            enc = base_tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            stop_tokens    = [base_tokenizer.eos_token_id]
            end_of_turn_id = base_tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if end_of_turn_id and end_of_turn_id != base_tokenizer.unk_token_id:
                stop_tokens.append(end_of_turn_id)

            ngram_size = 5 if winning_domain == "math" else 3

            streamer = TextIteratorStreamer(
                base_tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=60.0,
            )

            gen_kwargs = dict(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                no_repeat_ngram_size=ngram_size,
                pad_token_id=base_tokenizer.pad_token_id,
                eos_token_id=stop_tokens,
                streamer=streamer,
            )

            gen_thread = threading.Thread(
                target=lambda: _safe_generate(model, gen_kwargs)
            )
            gen_thread.start()

            full_response = ""
            for chunk in streamer:
                if chunk:
                    full_response += chunk
                    yield {"type": "token", "text": chunk}

            gen_thread.join()
        finally:
            cleanup()

    cleaned = _postprocess_response(full_response, winning_domain)
    t_total = time.time() - t_start

    yield {
        "type":          "done",
        "time_seconds":  round(t_total, 2),
        "full_response": cleaned,
    }


def _safe_generate(model, gen_kwargs):
    try:
        with torch.inference_mode():
            model.generate(**gen_kwargs)
    except Exception as e:
        print(f"[Stream] generate() raised: {e}")


if __name__ == "__main__":
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except ImportError:
        pass
    prepare()
    load_all_models()