from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/store-transcript', methods=['POST'])
def store_transcript_endpoint():
    data = request.get_json()
    video_url = data.get('video_url')

    if not video_url:
        return jsonify({"error": "video_url required"}), 400

    try:
        # Download
        print(f"📥 Downloading: {video_url}")
        video_path = download_video(video_url)

        # Transcribe
        print("🎙️ Transcribing...")
        transcript = get_transcript(video_path)

        # Summarize
        print("📝 Generating summary...")
        summary = generate_summary(transcript)

        # Store
        video_id = vector_store.add_video(video_url, transcript, video_path, summary)

        return jsonify({"video_id": video_id, "summary": summary, "status": "success"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query_endpoint():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "query required"}), 400

    try:
        # Search
        segments = vector_store.search(query, top_k=3)

        if not segments:
            return jsonify({"answer": "No videos indexed yet.", "video_url": None})

        # Get best video
        video_groups = {}
        for seg in segments:
            vid = seg["video_id"]
            video_groups.setdefault(vid, []).append(seg)

        best_video_id = max(video_groups, key=lambda v: sum(s["similarity"] for s in video_groups[v]))
        best_segments = video_groups[best_video_id]

        # Get video info
        video_url = vector_store.get_video_url(best_video_id)
        summary = vector_store.get_video_summary(best_video_id)

        # Timestamp
        timestamp = int(min(s["start"] for s in best_segments))
        timestamped_url = f"{video_url}&t={timestamp}" if "?" in video_url else f"{video_url}?t={timestamp}"

        # Generate answer
        answer = generate_response(query, summary or "", best_segments)

        return jsonify({
            "answer": answer,
            "video_url": timestamped_url,
            "timestamp": timestamp,
            "segments_used": len(best_segments)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats_endpoint():
    return jsonify(vector_store.get_stats())


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})


print("✅ Flask app ready")


from pyngrok import ngrok
import threading

# Set ngrok auth token
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Start Flask in a thread
def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

# Create ngrok tunnel
public_url = ngrok.connect(5000)
print("="*60)
print(f"🌐 PUBLIC URL: {public_url}")
print("="*60)
print("\nEndpoints:")
print(f"  POST {public_url}/store-transcript")
print(f"  POST {public_url}/query")
print(f"  GET  {public_url}/stats")
print(f"  GET  {public_url}/health")