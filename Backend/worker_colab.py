# ==============================================================================
# worker_colab.py — Run this ENTIRE file in a single Google Colab cell.
# Uses Flask (sync) + pyngrok to expose the AdaptRoute pipeline.
#
# Endpoints:
#   GET  /health
#   POST /generate          body: { query | messages, mode }   (non-streaming)
#   POST /generate-stream   body: { query | messages }         (SSE streaming)
#
# /generate modes: "routed" (default), "base", "both".
# /generate-stream always uses the routed path and emits events:
#   { "type": "meta",    "adapter_used": "...", "gating_scores": {...}, ... }
#   { "type": "token",   "text": "..." }
#   { "type": "done",    "seconds": 3.12, "total_chars": 420 }
#   { "type": "blocked", "message": "...", "firewall_label": "INJECTION" }
#   { "type": "error",   "message": "..." }
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
import json
import time
import threading
import torch
from threading import Thread
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from pyngrok import ngrok
from google.colab import userdata

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TextIteratorStreamer,
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

# Serialize generation — one model on one GPU, concurrent requests would collide
GEN_LOCK = threading.Lock()


# ==============================================================================
# PIPELINE — SETUP
# ==============================================================================
def prepare():
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

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    global_systems["base_tokenizer"] = tok
    global_systems["base_model"] = base_model
    global_systems["base_model"].eval()
    print("All models loaded.")


# ==============================================================================
# PIPELINE — HELPERS
# ==============================================================================
def _firewall_check(query: str) -> str:
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


def _format_prompt(messages_or_query) -> str:
    """
    Accepts either a string (single-shot) or a list of {role, content}.
    Uses Qwen's built-in chat template when given messages.
    """
    tokenizer = global_systems["base_tokenizer"]
    if isinstance(messages_or_query, str):
        messages = [{"role": "user", "content": messages_or_query}]
    else:
        messages = messages_or_query

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback to manual ChatML if tokenizer has no chat template
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


def _stop_token_ids():
    tokenizer = global_systems["base_tokenizer"]
    stop_tokens = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        stop_tokens.append(im_end_id)
    return stop_tokens


def _generate(query_or_messages, adapter_name):
    """Non-streaming generation. Returns (response_text, seconds, tokens)."""
    base_model = global_systems["base_model"]
    base_tokenizer = global_systems["base_tokenizer"]

    if isinstance(base_model, PeftModel):
        if adapter_name is None:
            base_model.disable_adapter_layers()
        else:
            base_model.enable_adapter_layers()
            base_model.set_adapter(adapter_name)

    try:
        prompt_text = _format_prompt(query_or_messages)
        enc = base_tokenizer(prompt_text, return_tensors="pt").to(base_model.device)

        t0 = time.time()
        with torch.no_grad():
            out = base_model.generate(
                **enc,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.5,
                pad_token_id=base_tokenizer.pad_token_id,
                eos_token_id=_stop_token_ids(),
            )
        elapsed = time.time() - t0

        response = base_tokenizer.decode(
            out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        tokens = out.shape[1] - enc["input_ids"].shape[1]
        return response, elapsed, tokens
    finally:
        if isinstance(base_model, PeftModel):
            base_model.enable_adapter_layers()


# ==============================================================================
# PIPELINE — ORCHESTRATION (non-streaming)
# ==============================================================================
def process_query(query_or_messages, mode: str = "routed") -> dict:
    if any(m is None for m in global_systems.values()):
        return {"status": "error", "message": "Models are not loaded."}

    if mode not in ("routed", "base", "both"):
        return {"status": "error", "message": f"Unknown mode: {mode}"}

    # Extract the user query to route on — if messages, use the latest user turn
    if isinstance(query_or_messages, str):
        query_text = query_or_messages
    else:
        user_msgs = [m for m in query_or_messages if m.get("role") == "user"]
        query_text = user_msgs[-1]["content"] if user_msgs else ""

    if not query_text.strip():
        return {"status": "error", "message": "No user query to process."}

    t_start = time.time()

    fw_label = _firewall_check(query_text)
    if fw_label == "INJECTION":
        return {
            "status": "blocked",
            "message": "Your query was flagged as a potential prompt injection attempt.",
            "firewall_label": fw_label,
            "mode": mode,
        }
    t_fw = time.time()

    winning_domain, gating_scores = _gate(query_text)
    if not winning_domain:
        return {
            "status": "error",
            "message": "Could not map gating network output to a known adapter.",
            "gating_scores": gating_scores,
        }
    t_gate = time.time()

    need_adapter = mode in ("routed", "both")
    if need_adapter:
        _ensure_adapter_loaded(winning_domain)
    t_adapter = time.time()

    result = {
        "status": "success",
        "mode": mode,
        "firewall_label": fw_label,
        "adapter_used": winning_domain,
        "gating_scores": gating_scores,
    }

    if mode == "routed":
        resp, el, tk = _generate(query_or_messages, adapter_name=winning_domain)
        result["response"] = resp
        result["generation_seconds"] = round(el, 2)
        result["tokens_generated"] = tk
    elif mode == "base":
        resp, el, tk = _generate(query_or_messages, adapter_name=None)
        result["response"] = resp
        result["generation_seconds"] = round(el, 2)
        result["tokens_generated"] = tk
    elif mode == "both":
        br, bel, btk = _generate(query_or_messages, adapter_name=None)
        rr, rel, rtk = _generate(query_or_messages, adapter_name=winning_domain)
        result.update({
            "base_response": br, "base_generation_seconds": round(bel, 2), "base_tokens_generated": btk,
            "routed_response": rr, "routed_generation_seconds": round(rel, 2), "routed_tokens_generated": rtk,
        })

    result["total_time_seconds"] = round(time.time() - t_start, 2)
    result["timing"] = {
        "firewall": round(t_fw - t_start, 3),
        "gating": round(t_gate - t_fw, 3),
        "adapter_load": round(t_adapter - t_gate, 3),
        "generation_total": round(time.time() - t_adapter, 3),
    }
    return result


# ==============================================================================
# PIPELINE — STREAMING
# ==============================================================================
def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def stream_query(messages):
    """
    Generator yielding SSE-formatted events for a chat-style request.
    Always uses the routed path.
    """
    # Latest user message drives firewall + gating
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        yield sse({"type": "error", "message": "No user message in conversation."})
        return
    latest_query = (user_msgs[-1].get("content") or "").strip()
    if not latest_query:
        yield sse({"type": "error", "message": "Latest user message is empty."})
        return

    try:
        # 1. Firewall
        fw_label = _firewall_check(latest_query)
        if fw_label == "INJECTION":
            yield sse({
                "type": "blocked",
                "message": "Your query was flagged as a potential prompt injection attempt.",
                "firewall_label": fw_label,
            })
            return

        # 2. Gating
        winning_domain, gating_scores = _gate(latest_query)
        if not winning_domain:
            yield sse({
                "type": "error",
                "message": "Could not route query to any known adapter.",
                "gating_scores": gating_scores,
            })
            return

        # 3. Load + activate adapter
        _ensure_adapter_loaded(winning_domain)
        base_model = global_systems["base_model"]
        base_tokenizer = global_systems["base_tokenizer"]
        base_model.enable_adapter_layers()
        base_model.set_adapter(winning_domain)

        # 4. Emit meta
        yield sse({
            "type": "meta",
            "adapter_used": winning_domain,
            "gating_scores": gating_scores,
            "firewall_label": fw_label,
        })

        # 5. Build prompt and streamer
        prompt_text = _format_prompt(messages)
        enc = base_tokenizer(prompt_text, return_tensors="pt").to(base_model.device)

        streamer = TextIteratorStreamer(
            base_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=120,
        )
        gen_kwargs = dict(
            **enc,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.5,
            pad_token_id=base_tokenizer.pad_token_id,
            eos_token_id=_stop_token_ids(),
            streamer=streamer,
        )

        t0 = time.time()
        gen_thread = Thread(target=base_model.generate, kwargs=gen_kwargs)
        gen_thread.start()

        total_chars = 0
        for new_text in streamer:
            if new_text:
                total_chars += len(new_text)
                yield sse({"type": "token", "text": new_text})

        gen_thread.join()
        yield sse({
            "type": "done",
            "seconds": round(time.time() - t0, 2),
            "total_chars": total_chars,
        })
    except Exception as e:
        yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})


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
    messages = data.get("messages")
    query = (data.get("query") or "").strip() if isinstance(data.get("query"), str) else ""
    mode = (data.get("mode") or "routed").strip().lower()

    if not messages and not query:
        return jsonify({"status": "error", "message": "Provide 'messages' or 'query'."}), 400

    payload = messages if messages else query

    with GEN_LOCK:
        result = process_query(payload, mode=mode)

    status_code = 200 if result.get("status") == "success" else (
        403 if result.get("status") == "blocked" else 500
    )
    return jsonify(result), status_code


@app.route("/generate-stream", methods=["POST"])
def generate_stream():
    data = request.get_json(force=True) or {}
    messages = data.get("messages")
    query = (data.get("query") or "").strip() if isinstance(data.get("query"), str) else ""

    if not messages and not query:
        return jsonify({"status": "error", "message": "Provide 'messages' or 'query'."}), 400

    if not messages:
        messages = [{"role": "user", "content": query}]

    def generator():
        # One request at a time — GPU can't safely run two generations concurrently
        with GEN_LOCK:
            for chunk in stream_query(messages):
                yield chunk

    return Response(
        generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ==============================================================================
# Step 4: Start Flask in background thread
# ==============================================================================
def run_flask():
    # threaded=True so streaming requests don't block health checks
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)


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
print(f"  POST {public_url}/generate          body: {{ query|messages, mode }}")
print(f"  POST {public_url}/generate-stream   body: {{ query|messages }}  (SSE)")
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