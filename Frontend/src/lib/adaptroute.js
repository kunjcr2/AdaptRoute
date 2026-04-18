// src/lib/adaptroute.js
// Thin client for the AdaptRoute worker. Simple JSON request/response.

const WORKER_URL =
  import.meta.env.VITE_WORKER_URL || "http://192.168.50.213:7180";

const defaultHeaders = {
  "Content-Type": "application/json",
  "ngrok-skip-browser-warning": "true",
};

export function getWorkerUrl() {
  return WORKER_URL;
}

export async function checkHealth() {
  if (!WORKER_URL) throw new Error("VITE_WORKER_URL not set");
  const res = await fetch(`${WORKER_URL}/health`, {
    headers: { "ngrok-skip-browser-warning": "true" },
  });
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

/**
 * Send a query and get response with routing metadata.
 * Returns: {
 *   status: "success" | "blocked" | "error",
 *   response: string (the generated response),
 *   adapter_used: string (code|math|qa|medical),
 *   gating_scores: { code: 0.xx, math: 0.xx, qa: 0.xx, medical: 0.xx },
 *   firewall_label: string (SAFE|INJECTION),
 *   time_seconds: number
 * }
 */
export async function sendQuery(query) {
  if (!WORKER_URL) throw new Error("VITE_WORKER_URL not set");

  const res = await fetch(`${WORKER_URL}/generate`, {
    method: "POST",
    headers: defaultHeaders,
    body: JSON.stringify({ query }),
  });

  if (!res.ok) {
    let msg = `Worker returned ${res.status}`;
    try {
      const body = await res.json();
      if (body?.message) msg = body.message;
    } catch (_) {}
    throw new Error(msg);
  }

  const data = await res.json();
  return data;
}

export async function getStats() {
  const res = await fetch(`${WORKER_URL}/stats`, { headers: defaultHeaders });
  if (!res.ok) throw new Error(`Stats failed: ${res.status}`);
  return res.json();
}

export async function getTrainStatus() {
  const res = await fetch(`${WORKER_URL}/train/status`, {
    headers: defaultHeaders,
  });
  if (!res.ok) throw new Error(`Train status failed: ${res.status}`);
  return res.json();
}

export async function startTraining() {
  const res = await fetch(`${WORKER_URL}/train`, {
    method: "POST",
    headers: defaultHeaders,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Train failed: ${res.status}`);
  }
  return res.json();
}

export async function stopTraining() {
  const res = await fetch(`${WORKER_URL}/train/stop`, {
    method: "POST",
    headers: defaultHeaders,
  });
  if (!res.ok) throw new Error(`Stop failed: ${res.status}`);
  return res.json();
}

/**
 * Stream a query response token-by-token.
 * Callbacks:
 *  - onMeta(obj): called with {adapter_used, routing_mode, ...}
 *  - onToken(text): called for each token in the response
 *  - onDone(obj): called when stream completes {time_seconds, ...}
 *  - onBlocked(message): called if request was blocked
 *  - onError(err): called on error
 */
export async function streamQuery(
  query,
  { onMeta, onToken, onDone, onBlocked, onError },
) {
  if (!WORKER_URL) throw new Error("VITE_WORKER_URL not set");

  const res = await fetch(`${WORKER_URL}/generate/stream`, {
    method: "POST",
    headers: defaultHeaders,
    body: JSON.stringify({ query }),
  });

  if (!res.ok) {
    onError?.(`HTTP ${res.status}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // last line may be incomplete — save for next chunk

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const obj = JSON.parse(line);
        if (obj.type === "meta") onMeta?.(obj);
        else if (obj.type === "token") onToken?.(obj.text);
        else if (obj.type === "done") onDone?.(obj);
        else if (obj.type === "blocked") onBlocked?.(obj.message);
        else if (obj.type === "error") onError?.(obj.message);
      } catch (e) {
        console.error("Bad JSON line:", line);
      }
    }
  }
}
