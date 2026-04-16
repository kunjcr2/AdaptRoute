# ==============================================================================
# worker_colab.py — Run this ENTIRE file in a single Google Colab cell.
# Uses Flask (sync, no nest_asyncio needed) + pyngrok to expose the pipeline.
# ==============================================================================

# ── Step 0: Install dependencies ──────────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "-q", "flask", "pyngrok",
    "transformers", "peft", "bitsandbytes", "accelerate",
    "datasets", "huggingface_hub"
], check=True)

import threading
import time
from flask import Flask, request, jsonify
from pyngrok import ngrok
from google.colab import userdata

# ── Step 1: Ngrok Auth ────────────────────────────────────────────────────────
NGROK_AUTH_TOKEN = userdata.get("NGROK")
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

print("==> Starting model loading...")
prepare()
load_all_models()
print("==> All models ready.")

# ── Step 3: Flask App ─────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "AdaptRoute worker is running."})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"status": "error", "message": "Query cannot be empty."}), 400

    result = process_query(query)
    return jsonify(result)

# ── Step 4: Start Flask in background thread ──────────────────────────────────
def run_flask():
    app.run(port=5000, use_reloader=False)

threading.Thread(target=run_flask, daemon=True).start()
time.sleep(2)  # Give Flask a moment to start

# ── Step 5: Open ngrok tunnel ─────────────────────────────────────────────────
public_url = ngrok.connect(5000).public_url

print("\n" + "="*60)
print("  AdaptRoute Worker is LIVE!")
print(f"  Public URL: {public_url}")
print("  Copy this URL into local_bridge.py → COLAB_URL")
print("="*60 + "\n")

# Keep cell alive
while True:
    time.sleep(60)
