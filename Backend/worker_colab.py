# ==============================================================================
# worker_colab.py — Run this ENTIRE file in a single Google Colab cell.
# Uses Flask (sync, no nest_asyncio needed) + pyngrok to expose the pipeline.
#
# /generate accepts: {"query": "...", "mode": "routed" | "base" | "both"}
#   - "routed" (default): firewall + gating + adapter + generate
#   - "base":   firewall + gating (for scores) + generate with NO adapter
#   - "both":   runs both base and routed in one request, returns both
# ==============================================================================

# ── Step 0: Install dependencies ──────────────────────────────────────────────
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "flask", "flask-cors", "pyngrok",
    "transformers", "peft", "bitsandbytes", "accelerate",
    "datasets", "huggingface_hub"
], check=True)

import os
import time
import threading
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from google.colab import userdata

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig
from huggingface_hub import snapshot_download


# ==============================================================================
# CONFIGURATION
# ==============================================================================
FIREWALL_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
GATING_MODEL = "kunjcr2/gating-bert-adaptroute"
BASE_MODEL = "Qwen/Qwen2.5-1.5B"

ADAPTER_REPOS = {
    "code": "kunjcr2/code-adaptroute-v3",
    "math": "kunjcr2/math-adaptroute-v3",
    "qa": "kunjcr2/qa-adaptroute-v3",
    "medical": "kunjcr2/medical-adaptroute-v3",
}

ADAPTERS_DIR = os.path.abspath(os.path.join(os.getcwd(), "Adapters"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ==============================================================================
# PIPELINE — SETUP
# ==============================================================================
def prepare():
    """Download adapters from HF Hub if they don't exist locally."""
    os.makedirs(ADAPTERS_DIR, exist_ok=True)
    existing_items = [
        name for name in os.listdir(ADAPTERS_DIR)
        if os.path.isdir(os.path.join(ADAPTERS_DIR, name))
    ]

    if not existing_items:
        print(f"Adapters folder is empty. Downloading adapters to {ADAPTERS_DIR}...")
        for domain, repo_id in ADAPTER_REPOS.items():
            local_path = os.path.join(ADAPTERS_DIR, domain)
            print(f"Downloading {repo_id} to {local_path}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_path,
                ignore_patterns=["*.msgpack", "*.h5"],
            )
        print("Finished downloading all adapters.")
    else:
        print(f"Adapters already exist locally in {ADAPTERS_DIR}.")


global_systems = {
    "firewall_model": None,
    "firewall_tokenizer": None,
    "gating_model": None,
    "gating_tokenizer": None,
    "base_model": None,
    "base_tokenizer": None,
}


def load_all_models():
    """Load the Firewall, Gating Network, and Base Model into RAM."""
    global global_systems

    print("Loading Firewall Model...")
    global_systems["firewall_tokenizer"] = AutoTokenizer.from_pretrained(FIREWALL_MODEL)
    global_systems["firewall_model"] = (
        AutoModelForSequenceClassification.from_pretrained(FIREWALL_MODEL).to(DEVICE)
    )
    global_systems["firewall_model"].eval()

    print("Loading Gating Network...")
    global_systems["gating_tokenizer"] = AutoTokenizer.from_pretrained(GATING_MODEL)
    global_systems["gating_model"] = (
        AutoModelForSequenceClassification.from_pretrained(GATING_MODEL).to(DEVICE)
    )
    global_systems["gating_model"].eval()

    print("Loading Base Model natively in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    base_model.config.use_cache = True

    global_systems["base_tokenizer"] = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True
    )
    if global_systems["base_tokenizer"].pad_token is None:
        global_systems["base_tokenizer"].pad_token = global_systems["base_tokenizer"].eos_token
    global_systems["base_tokenizer"].padding_side = "right"

    global_systems["base_model"] = base_model
    global_systems["base_model"].eval()

    print("All models loaded successfully! (Adapters deferred until queried)")


# ==============================================================================
# PIPELINE — HELPERS
# ==============================================================================
def _firewall_check(query: str) -> str:
    """Returns the firewall label, e.g. 'SAFE' or 'INJECTION'."""
    fw_tokenizer = global_systems["firewall_tokenizer"]
    fw_model = global_systems["firewall_model"]
    fw_inputs = fw_tokenizer(
        query, return_tensors="pt", truncation=True, max_length=512
    ).to(fw_model.device)
    with torch.no_grad():
        fw_outputs = fw_model(**fw_inputs)
    predicted = fw_outputs.logits.argmax(dim=-1).item()
    return fw_model.config.id2label.get(predicted, "SAFE")


def _gate(query: str):
    """Returns (winning_domain, gating_scores_dict)."""
    gating_tokenizer = global_systems["gating_tokenizer"]
    gating_model = global_systems["gating_model"]
    gate_inputs = gating_tokenizer(
        query, return_tensors="pt", truncation=True, max_length=512
    ).to(gating_model.device)
    with torch.no_grad():
        gate_outputs = gating_model(**gate_inputs)

    probs = torch.softmax(gate_outputs.logits, dim=-1).squeeze()
    best_idx = probs.argmax().item()
    gate_id2label = gating_model.config.id2label
    winning_label = gate_id2label.get(best_idx, "").lower()

    winning_domain = None
    for domain in ADAPTER_REPOS.keys():
        if domain.lower() in winning_label:
            winning_domain = domain
            break

    gating_scores = {
        gate_id2label.get(i, str(i)).lower(): round(probs[i].item(), 4)
        for i in range(len(probs))
    }
    return winning_domain, gating_scores


def _ensure_adapter_loaded(winning_domain: str):
    """Load the adapter into the PEFT wrapper if not already present."""
    base_model = global_systems["base_model"]
    local_adapter_path = os.path.join(ADAPTERS_DIR, winning_domain)

    if not isinstance(base_model, PeftModel):
        base_model = PeftModel.from_pretrained(
            base_model, local_adapter_path, adapter_name=winning_domain
        )
        global_systems["base_model"] = base_model
    else:
        if winning_domain not in base_model.peft_config:
            base_model.load_adapter(local_adapter_path, adapter_name=winning_domain)


def _generate(query: str, adapter_name: str | None) -> tuple[str, float, int]:
    """
    Generate a response from the base model.
    If adapter_name is None, disables adapters (true base model output).
    Returns (response_text, seconds_elapsed, tokens_generated).
    """
    base_model = global_systems["base_model"]
    base_tokenizer = global_systems["base_tokenizer"]

    # Switch adapter state
    if isinstance(base_model, PeftModel):
        if adapter_name is None:
            # Disable all adapters → pure base model
            base_model.disable_adapter_layers()
        else:
            base_model.enable_adapter_layers()
            base_model.set_adapter(adapter_name)

    try:
        formatted_prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        enc = base_tokenizer(formatted_prompt, return_tensors="pt").to(base_model.device)

        stop_tokens = [base_tokenizer.eos_token_id]
        im_end_id = base_tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != base_tokenizer.unk_token_id:
            stop_tokens.append(im_end_id)

        t0 = time.time()
        with torch.no_grad():
            out = base_model.generate(
                **enc,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.5,
                pad_token_id=base_tokenizer.pad_token_id,
                eos_token_id=stop_tokens,
            )
        elapsed = time.time() - t0

        response = base_tokenizer.decode(
            out[0][enc["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        generated_tokens = out.shape[1] - enc["input_ids"].shape[1]
        return response, elapsed, generated_tokens
    finally:
        # Re-enable adapter layers so subsequent calls default to routed behavior
        if isinstance(base_model, PeftModel):
            base_model.enable_adapter_layers()


# ==============================================================================
# PIPELINE — ORCHESTRATION
# ==============================================================================
def process_query(query: str, mode: str = "routed") -> dict:
    """
    mode:
      "routed" → firewall + gating + adapter + generate (single response)
      "base"   → firewall + gating (scores only) + generate with NO adapter
      "both"   → both base and routed responses returned together
    """
    if any(m is None for m in global_systems.values()):
        return {"status": "error", "message": "Models are not loaded."}

    if mode not in ("routed", "base", "both"):
        return {"status": "error", "message": f"Unknown mode: {mode}"}

    t_start = time.time()

    # 1. Firewall
    fw_label = _firewall_check(query)
    if fw_label == "INJECTION":
        return {
            "status": "blocked",
            "message": (
                "Your query was flagged as a potential prompt injection attempt "
                "and could not be processed. Please rephrase your request."
            ),
            "firewall_label": fw_label,
            "mode": mode,
        }
    t_fw = time.time()

    # 2. Gating
    winning_domain, gating_scores = _gate(query)
    if not winning_domain:
        return {
            "status": "error",
            "message": "Could not map gating network output to a known adapter.",
            "gating_scores": gating_scores,
        }
    t_gate = time.time()

    # 3. Ensure adapter is loaded (needed for routed or both; harmless for base)
    need_adapter = mode in ("routed", "both")
    if need_adapter:
        _ensure_adapter_loaded(winning_domain)
    t_adapter = time.time()

    # 4. Generate per mode
    result = {
        "status": "success",
        "mode": mode,
        "firewall_label": fw_label,
        "adapter_used": winning_domain,
        "gating_scores": gating_scores,
    }

    if mode == "routed":
        resp, elapsed, tokens = _generate(query, adapter_name=winning_domain)
        result["response"] = resp
        result["generation_seconds"] = round(elapsed, 2)
        result["tokens_generated"] = tokens

    elif mode == "base":
        resp, elapsed, tokens = _generate(query, adapter_name=None)
        result["response"] = resp
        result["generation_seconds"] = round(elapsed, 2)
        result["tokens_generated"] = tokens

    elif mode == "both":
        base_resp, base_elapsed, base_tokens = _generate(query, adapter_name=None)
        routed_resp, routed_elapsed, routed_tokens = _generate(query, adapter_name=winning_domain)
        result["base_response"] = base_resp
        result["base_generation_seconds"] = round(base_elapsed, 2)
        result["base_tokens_generated"] = base_tokens
        result["routed_response"] = routed_resp
        result["routed_generation_seconds"] = round(routed_elapsed, 2)
        result["routed_tokens_generated"] = routed_tokens

    t_total = time.time() - t_start
    result["total_time_seconds"] = round(t_total, 2)
    result["timing"] = {
        "firewall": round(t_fw - t_start, 3),
        "gating": round(t_gate - t_fw, 3),
        "adapter_load": round(t_adapter - t_gate, 3),
        "generation_total": round(time.time() - t_adapter, 3),
    }

    return result


# ==============================================================================
# Step 1: Ngrok Auth
# ==============================================================================
NGROK_AUTH_TOKEN = userdata.get("NGROK")
ngrok.set_auth_token(NGROK_AUTH_TOKEN)


# ==============================================================================
# Step 2: Model loading
# ==============================================================================
print("==> Starting model loading...")
prepare()
load_all_models()
print("==> All models ready.")


# ==============================================================================
# Step 3: Flask App
# ==============================================================================
app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "AdaptRoute worker is running."})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    mode = (data.get("mode") or "routed").strip().lower()

    if not query:
        return jsonify({"status": "error", "message": "Query cannot be empty."}), 400

    result = process_query(query, mode=mode)
    status_code = 200 if result.get("status") == "success" else (
        403 if result.get("status") == "blocked" else 500
    )
    return jsonify(result), status_code


# ==============================================================================
# Step 4: Start Flask in background thread
# ==============================================================================
def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
time.sleep(2)


# ==============================================================================
# Step 5: Open ngrok tunnel
# ==============================================================================
public_url = ngrok.connect(5000).public_url

print("\n" + "=" * 60)
print("  AdaptRoute Worker is LIVE!")
print(f"  Public URL: {public_url}")
print("  Paste into your frontend .env as VITE_WORKER_URL")
print("=" * 60)
print("\nEndpoints:")
print(f"  GET  {public_url}/health")
print(f"  POST {public_url}/generate   body: {{ query, mode }}")
print()


# ==============================================================================
# Keep cell alive
# ==============================================================================
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("Shutting down...")
    ngrok.disconnect(public_url)
    ngrok.kill()