// src/lib/adaptroute.js
// Thin client for the AdaptRoute worker. Handles SSE streaming over POST.

const WORKER_URL = import.meta.env.VITE_WORKER_URL || '';

const defaultHeaders = {
    'Content-Type': 'application/json',
    'ngrok-skip-browser-warning': 'true',
};

export function getWorkerUrl() {
    return WORKER_URL;
}

export async function checkHealth() {
    if (!WORKER_URL) throw new Error('VITE_WORKER_URL not set');
    const res = await fetch(`${WORKER_URL}/health`, {
        headers: { 'ngrok-skip-browser-warning': 'true' },
    });
    if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
    return res.json();
}

/**
 * Stream a chat completion. Emits events via onEvent:
 *   { type: 'meta',    adapter_used, gating_scores, firewall_label }
 *   { type: 'token',   text }
 *   { type: 'done',    seconds, total_chars }
 *   { type: 'blocked', message, firewall_label }
 *   { type: 'error',   message }
 *
 * messages: [{ role: 'user' | 'assistant', content: string }]
 * signal:   optional AbortSignal for stop button
 */
export async function streamChat({ messages, onEvent, signal }) {
    if (!WORKER_URL) throw new Error('VITE_WORKER_URL not set');

    const res = await fetch(`${WORKER_URL}/generate-stream`, {
        method: 'POST',
        headers: defaultHeaders,
        body: JSON.stringify({ messages }),
        signal,
    });

    if (!res.ok) {
        let msg = `Worker returned ${res.status}`;
        try {
            const body = await res.json();
            if (body?.message) msg = body.message;
        } catch (_) { }
        throw new Error(msg);
    }

    if (!res.body) throw new Error('No response body (streaming unsupported?)');

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by a blank line
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
            const line = part.trim();
            if (!line.startsWith('data:')) continue;
            const payload = line.slice(5).trim();
            if (!payload) continue;
            try {
                const event = JSON.parse(payload);
                onEvent(event);
            } catch (e) {
                console.warn('Failed to parse SSE event', payload, e);
            }
        }
    }
}