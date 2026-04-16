# ==============================================================================
# COLAB WORKER — Expose pipeline via Flask + ngrok
# Append this to the same cell as your pipeline.py
# ==============================================================================

# Install deps
import subprocess
subprocess.run(["pip", "install", "-q", "flask", "flask-cors", "pyngrok"], check=True)

import json, time, threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from google.colab import userdata

# ==============================================================================
# Configure ngrok
# ==============================================================================
NGROK_TOKEN = userdata.get("NGROK")
ngrok.set_auth_token(NGROK_TOKEN)

# ==============================================================================
# Flask App
# ==============================================================================
app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Pipeline running."})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    
    if not query:
        return jsonify({"status": "error", "message": "Provide 'query' in request body."}), 400
    
    result = process_query(query)
    return jsonify(result), 200

# ==============================================================================
# Start Flask & ngrok
# ==============================================================================
def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
time.sleep(2)

public_url = ngrok.connect(5000).public_url

print("\n" + "=" * 60)
print(f"✓ Pipeline exposed at: {public_url}")
print("=" * 60)
print(f"POST {public_url}/generate  (body: {{'query': '...'}})")
print("=" * 60)

# Keep cell alive
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("Shutting down...")
    ngrok.disconnect(public_url)
    ngrok.kill()