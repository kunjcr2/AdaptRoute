// src/lib/adaptroute.js
// Thin client for the AdaptRoute worker. Simple JSON request/response.

const WORKER_URL = import.meta.env.VITE_WORKER_URL || "https://swimsuit-threaten-resubmit.ngrok-free.dev";

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
