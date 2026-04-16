import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Loader2, AlertTriangle, ShieldAlert, Cpu, Combine, Clock, Hash } from 'lucide-react';

const WORKER_URL = import.meta.env.VITE_WORKER_URL || '';

const EXAMPLE_QUERIES = [
    { label: 'Code', text: 'Write a Python function that returns the nth Fibonacci number using memoization.' },
    { label: 'Math', text: 'If f(x) = 3x^2 + 2x - 5, what is f\'(2)?' },
    { label: 'QA', text: 'Who wrote the novel "One Hundred Years of Solitude" and in what year?' },
    { label: 'Medical', text: 'What are the common early symptoms of type 2 diabetes?' },
];

const Demo = () => {
    const [query, setQuery] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [workerStatus, setWorkerStatus] = useState('checking'); // checking | online | offline

    // Health check on mount
    useEffect(() => {
        if (!WORKER_URL) {
            setWorkerStatus('offline');
            return;
        }
        let cancelled = false;
        fetch(`${WORKER_URL}/health`, {
            headers: { 'ngrok-skip-browser-warning': 'true' },
        })
            .then((r) => (r.ok ? r.json() : Promise.reject(r)))
            .then(() => !cancelled && setWorkerStatus('online'))
            .catch(() => !cancelled && setWorkerStatus('offline'));
        return () => { cancelled = true; };
    }, []);

    const runDemo = async () => {
        if (!query.trim()) return;
        if (!WORKER_URL) {
            setError('VITE_WORKER_URL is not set. Add it to your .env and restart the dev server.');
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const res = await fetch(`${WORKER_URL}/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'ngrok-skip-browser-warning': 'true',
                },
                body: JSON.stringify({ query: query.trim(), mode: 'both' }),
            });

            const data = await res.json();

            if (data.status === 'blocked') {
                setError(data.message || 'Query blocked by firewall.');
                setResult(data);
            } else if (data.status === 'error' || !res.ok) {
                setError(data.message || `Request failed (${res.status})`);
            } else {
                setResult(data);
            }
        } catch (e) {
            setError(`Network error: ${e.message}. Is the worker running and is VITE_WORKER_URL correct?`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container mx-auto px-6 py-20 max-w-6xl">
            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-12 text-center"
            >
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-brand-200 text-brand-800 text-xs font-semibold uppercase tracking-wider mb-6 shadow-sm">
                    <span className={`w-2 h-2 rounded-full ${workerStatus === 'online' ? 'bg-green-500 animate-pulse' :
                            workerStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500 animate-pulse'
                        }`}></span>
                    Worker {workerStatus}
                </div>
                <h1 className="font-serif text-5xl font-bold mb-4 text-brand-900">Live Demo</h1>
                <p className="text-xl text-brand-600 font-light max-w-2xl mx-auto">
                    Compare the base Qwen2.5-1.5B model against the same model with AdaptRoute's dynamic
                    LoRA routing — side by side, on your query.
                </p>
            </motion.div>

            {/* Input Card */}
            <div className="bg-white p-8 rounded-3xl border border-brand-100 shadow-xl mb-8">
                <label className="text-xs font-semibold text-brand-500 uppercase tracking-wider mb-3 block">
                    Your query
                </label>
                <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Ask anything — code, math, general knowledge, or medical..."
                    rows={4}
                    className="w-full p-4 rounded-xl border border-brand-200 bg-brand-50/30 text-brand-900 placeholder-brand-400 focus:outline-none focus:border-brand-500 focus:bg-white transition-all resize-none font-mono text-sm"
                    disabled={loading}
                />

                {/* Example chips */}
                <div className="flex flex-wrap gap-2 mt-4">
                    <span className="text-xs text-brand-500 self-center mr-2">Try:</span>
                    {EXAMPLE_QUERIES.map((ex) => (
                        <button
                            key={ex.label}
                            onClick={() => setQuery(ex.text)}
                            disabled={loading}
                            className="px-3 py-1 text-xs bg-brand-50 border border-brand-200 rounded-full text-brand-700 hover:bg-brand-100 transition-colors disabled:opacity-50"
                        >
                            {ex.label}
                        </button>
                    ))}
                </div>

                {/* Run button */}
                <div className="flex justify-end mt-6">
                    <button
                        onClick={runDemo}
                        disabled={loading || !query.trim() || workerStatus !== 'online'}
                        className="flex items-center gap-2 bg-brand-900 text-white px-6 py-3 rounded-full font-medium hover:bg-brand-800 transition-all shadow-lg disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                        {loading ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Generating...
                            </>
                        ) : (
                            <>
                                <Play className="w-4 h-4" />
                                Run Comparison
                            </>
                        )}
                    </button>
                </div>
            </div>

            {/* Error display */}
            <AnimatePresence>
                {error && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="mb-8 bg-red-50 border border-red-200 rounded-2xl p-5 flex gap-4"
                    >
                        {result?.status === 'blocked' ? (
                            <ShieldAlert className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
                        ) : (
                            <AlertTriangle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
                        )}
                        <div className="text-sm text-red-800">{error}</div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Results */}
            <AnimatePresence>
                {result && result.status === 'success' && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="space-y-6"
                    >
                        {/* Metadata bar */}
                        <div className="bg-brand-900 text-white rounded-2xl p-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                            <MetaStat icon={<Combine className="w-4 h-4" />} label="Routed to" value={result.adapter_used} />
                            <MetaStat icon={<Clock className="w-4 h-4" />} label="Total time" value={`${result.total_time_seconds}s`} />
                            <MetaStat icon={<Hash className="w-4 h-4" />} label="Base tokens" value={result.base_tokens_generated} />
                            <MetaStat icon={<Hash className="w-4 h-4" />} label="Routed tokens" value={result.routed_tokens_generated} />
                        </div>

                        {/* Gating scores */}
                        {result.gating_scores && (
                            <div className="bg-white rounded-2xl border border-brand-100 p-6 shadow-sm">
                                <div className="text-xs font-semibold text-brand-500 uppercase tracking-wider mb-3">
                                    Gating Network Confidence
                                </div>
                                <div className="space-y-2">
                                    {Object.entries(result.gating_scores)
                                        .sort(([, a], [, b]) => b - a)
                                        .map(([label, score]) => (
                                            <div key={label} className="flex items-center gap-3">
                                                <div className="w-20 text-xs font-medium text-brand-700 capitalize">{label}</div>
                                                <div className="flex-1 h-2 bg-brand-50 rounded-full overflow-hidden">
                                                    <div
                                                        className={`h-full rounded-full ${label.toLowerCase().includes(result.adapter_used) ? 'bg-brand-600' : 'bg-brand-300'
                                                            }`}
                                                        style={{ width: `${Math.max(score * 100, 2)}%` }}
                                                    />
                                                </div>
                                                <div className="w-14 text-xs font-mono text-brand-600 text-right">
                                                    {(score * 100).toFixed(1)}%
                                                </div>
                                            </div>
                                        ))}
                                </div>
                            </div>
                        )}

                        {/* Side-by-side responses */}
                        <div className="grid md:grid-cols-2 gap-6">
                            <ResponseCard
                                title="Base Model"
                                subtitle="Qwen2.5-1.5B, no adapter"
                                icon={<Cpu className="w-5 h-5" />}
                                response={result.base_response}
                                seconds={result.base_generation_seconds}
                                tokens={result.base_tokens_generated}
                                accent="gray"
                            />
                            <ResponseCard
                                title="AdaptRoute"
                                subtitle={`Qwen2.5-1.5B + ${result.adapter_used} adapter`}
                                icon={<Combine className="w-5 h-5" />}
                                response={result.routed_response}
                                seconds={result.routed_generation_seconds}
                                tokens={result.routed_tokens_generated}
                                accent="brand"
                            />
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Offline helper */}
            {workerStatus === 'offline' && !loading && (
                <div className="mt-12 bg-yellow-50 border border-yellow-200 rounded-2xl p-6 text-sm text-yellow-900">
                    <div className="font-semibold mb-2">Worker is not reachable.</div>
                    <div className="text-yellow-800">
                        Start the Colab worker (<code className="bg-yellow-100 px-1 rounded">worker_colab.py</code>),
                        copy the ngrok URL it prints, and set{' '}
                        <code className="bg-yellow-100 px-1 rounded">VITE_WORKER_URL</code> in your{' '}
                        <code className="bg-yellow-100 px-1 rounded">.env</code> file.
                        Restart the dev server after changing .env.
                    </div>
                </div>
            )}
        </div>
    );
};

const MetaStat = ({ icon, label, value }) => (
    <div>
        <div className="flex items-center gap-1.5 text-brand-300 text-xs uppercase tracking-wider mb-1">
            {icon}
            {label}
        </div>
        <div className="font-mono text-lg font-semibold">{value ?? '—'}</div>
    </div>
);

const ResponseCard = ({ title, subtitle, icon, response, seconds, tokens, accent }) => {
    const isBrand = accent === 'brand';
    return (
        <div className={`rounded-3xl border shadow-xl overflow-hidden ${isBrand ? 'bg-white border-brand-200' : 'bg-gray-50 border-gray-200'
            }`}>
            <div className={`p-5 border-b flex items-center gap-3 ${isBrand ? 'bg-brand-50 border-brand-100 text-brand-800' : 'bg-gray-100 border-gray-200 text-gray-700'
                }`}>
                <div className={`p-2 rounded-lg shadow-sm border ${isBrand ? 'bg-white border-brand-100' : 'bg-white border-gray-200'
                    }`}>
                    {icon}
                </div>
                <div>
                    <div className="font-bold text-sm">{title}</div>
                    <div className="text-xs opacity-75">{subtitle}</div>
                </div>
            </div>
            <div className="p-6">
                <div className="text-sm text-brand-900 whitespace-pre-wrap leading-relaxed min-h-[8rem]">
                    {response || <span className="text-brand-400 italic">No response</span>}
                </div>
                <div className="mt-6 pt-4 border-t border-gray-200 flex gap-6 text-xs text-brand-500">
                    <span><Clock className="w-3 h-3 inline mr-1" />{seconds}s</span>
                    <span><Hash className="w-3 h-3 inline mr-1" />{tokens} tokens</span>
                </div>
            </div>
        </div>
    );
};

export default Demo;