import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Send, Plus, Trash2, MessageSquare, Copy, RefreshCw,
    Loader2, Cpu, ShieldAlert, AlertTriangle, User, Sparkles, Check,
} from 'lucide-react';
import { streamQuery, checkHealth, getWorkerUrl, getStats } from '../lib/adaptroute';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const STORAGE_KEY = 'adaptroute-chats-v1';

// ── Chat storage ────────────────────────────────────────────────────────────
const loadChats = () => {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed : [];
    } catch {
        return [];
    }
};

const saveChats = (chats) => {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
    } catch (e) {
        console.warn('Failed to save chats', e);
    }
};

const newChat = () => ({
    id: `c_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    title: 'New chat',
    messages: [],
    createdAt: Date.now(),
});

const titleFromMessage = (text) => {
    const t = (text || '').trim().split('\n')[0];
    return t.length > 40 ? `${t.slice(0, 40)}…` : (t || 'New chat');
};

// ── Main component ──────────────────────────────────────────────────────────
const Demo = () => {
    const [chats, setChats] = useState([]);
    const [activeId, setActiveId] = useState(null);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [workerStatus, setWorkerStatus] = useState('checking');
    const [queryCount, setQueryCount] = useState(null);
    const messagesEndRef = useRef(null);

    // Load chats + health check on mount
    useEffect(() => {
        const loaded = loadChats();
        if (loaded.length === 0) {
            const c = newChat();
            setChats([c]);
            setActiveId(c.id);
        } else {
            setChats(loaded);
            setActiveId(loaded[0].id);
        }
    }, []);

    useEffect(() => {
        let cancelled = false;
        if (!getWorkerUrl()) {
            setWorkerStatus('offline');
            return;
        }
        checkHealth()
            .then(() => !cancelled && setWorkerStatus('online'))
            .catch(() => !cancelled && setWorkerStatus('offline'));
        const interval = setInterval(() => {
            checkHealth()
                .then(() => !cancelled && setWorkerStatus('online'))
                .catch(() => !cancelled && setWorkerStatus('offline'));
        }, 30000);
        return () => { cancelled = true; clearInterval(interval); };
    }, []);

    // Fetch query count on mount and after each completed query
    const refreshStats = useCallback(() => {
        getStats().then((s) => setQueryCount(s.query_count)).catch(() => { });
    }, []);

    useEffect(() => { refreshStats(); }, [refreshStats]);

    // Persist chats
    useEffect(() => {
        if (chats.length > 0) saveChats(chats);
    }, [chats]);

    const activeChat = useMemo(
        () => chats.find((c) => c.id === activeId) || null,
        [chats, activeId]
    );
    // Latest assistant message in active chat that has routing metadata
    const latestRouting = useMemo(() => {
        if (!activeChat) return null;
        for (let i = activeChat.messages.length - 1; i >= 0; i--) {
            const m = activeChat.messages[i];
            if (m.role === 'assistant' && m.gatingScores && !m.blocked) {
                return m;
            }
        }
        return null;
    }, [activeChat]);

    // Track message count so we only scroll when a NEW message is added,
    // not when the last message's content is being updated by streaming tokens.
    const prevMessageCountRef = useRef(0);

    useEffect(() => {
        const count = activeChat?.messages.length ?? 0;
        if (count > prevMessageCountRef.current) {
            messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
        }
        prevMessageCountRef.current = count;
    }, [activeChat?.messages.length, activeChat?.id]);

    // ── Chat management ──
    const createChat = () => {
        const c = newChat();
        setChats((prev) => [c, ...prev]);
        setActiveId(c.id);
        setInput('');
        setError(null);
    };

    const deleteChat = (id) => {
        setChats((prev) => {
            const next = prev.filter((c) => c.id !== id);
            if (next.length === 0) {
                const c = newChat();
                setActiveId(c.id);
                return [c];
            }
            if (id === activeId) setActiveId(next[0].id);
            return next;
        });
    };

    const updateActiveChat = useCallback((updater) => {
        setChats((prev) => prev.map((c) => (c.id === activeId ? updater(c) : c)));
    }, [activeId]);

    // Helper: mutate only the LAST assistant message of the active chat.
    // Used heavily during streaming to append tokens without touching earlier messages.
    const patchLastAssistant = useCallback((patch) => {
        updateActiveChat((c) => {
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            if (!last || last.role !== 'assistant') return c;
            msgs[msgs.length - 1] = typeof patch === 'function' ? patch(last) : { ...last, ...patch };
            return { ...c, messages: msgs };
        });
    }, [updateActiveChat]);

    // ── Send query (STREAMING) ──
    const runQuery = useCallback(async (query) => {
        if (workerStatus !== 'online') {
            setError('Worker is not reachable. Start the Colab cell and check VITE_WORKER_URL.');
            return;
        }
        setError(null);
        setLoading(true);

        // Insert a placeholder assistant message that we'll mutate as tokens stream in.
        const placeholder = {
            role: 'assistant',
            content: '',
            adapter: null,
            routingMode: null,
            confidence: null,
            gatingScores: null,
            firewallLabel: null,
            blocked: false,
            streaming: true,
        };
        updateActiveChat((c) => ({ ...c, messages: [...c.messages, placeholder] }));

        try {
            await streamQuery(query, {
                // Meta arrives first: routing info, gating scores, firewall label.
                // Show all the badges immediately so the user sees routing before tokens arrive.
                onMeta: (meta) => {
                    patchLastAssistant({
                        adapter: meta.adapter_used,
                        routingMode: meta.routing_mode,
                        confidence: meta.gating_confidence,
                        gatingScores: meta.gating_scores,
                        firewallLabel: meta.firewall_label,
                    });
                },

                // Each token chunk is appended to the running content string.
                // Using the functional form of patchLastAssistant so we always
                // append to the latest content, not a stale snapshot.
                onToken: (text) => {
                    patchLastAssistant((last) => ({
                        ...last,
                        content: (last.content || '') + text,
                    }));
                },

                // Stream completed cleanly. Swap in the server's post-processed
                // final response (trailing cleanup, etc.) and stamp the timing.
                onDone: (info) => {
                    patchLastAssistant((last) => ({
                        ...last,
                        // content: info.full_response ?? last.content,
                        seconds: info.time_seconds,
                        streaming: false,
                    }));
                },

                // Firewall blocked the query — mark the placeholder as blocked.
                onBlocked: (message) => {
                    patchLastAssistant({
                        blocked: true,
                        streaming: false,
                        content: message || 'Query blocked by firewall.',
                        firewallLabel: 'INJECTION',
                    });
                },

                // Stream-level error. Mark the bubble as errored.
                onError: (err) => {
                    setError(typeof err === 'string' ? err : (err?.message || 'Stream error'));
                    patchLastAssistant((last) => ({
                        ...last,
                        error: true,
                        streaming: false,
                        content: last.content || 'Worker error.',
                    }));
                },
            });
        } catch (e) {
            // Network-level failure before the stream even started.
            setError(e.message || String(e));
            updateActiveChat((c) => ({
                ...c,
                messages: c.messages.slice(0, -1), // drop the empty placeholder
            }));
        } finally {
            setLoading(false);
            refreshStats();
        }
    }, [updateActiveChat, patchLastAssistant, workerStatus, refreshStats]);

    const sendMessage = async () => {
        const text = input.trim();
        if (!text || loading || !activeChat) return;

        const userMsg = { role: 'user', content: text };
        const isFirst = activeChat.messages.length === 0;
        updateActiveChat((c) => ({
            ...c,
            title: isFirst ? titleFromMessage(text) : c.title,
            messages: [...c.messages, userMsg],
        }));
        setInput('');

        await runQuery(text);
    };

    const regenerate = async () => {
        if (!activeChat || loading) return;
        const msgs = [...activeChat.messages];
        while (msgs.length > 0 && msgs[msgs.length - 1].role === 'assistant') {
            msgs.pop();
        }
        if (msgs.length === 0) return;

        const lastUserMsg = msgs[msgs.length - 1];
        if (lastUserMsg?.role !== 'user') return;

        updateActiveChat((c) => ({ ...c, messages: msgs }));
        await runQuery(lastUserMsg.content);
    };

    const onKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const canRegenerate = activeChat?.messages.some((m) => m.role === 'assistant') && !loading;

    // ── Render ──
    return (
        <div className="flex h-[calc(100vh-4rem)] bg-brand-50/30">
            {/* Sidebar */}
            <aside className="w-72 shrink-0 bg-white border-r border-brand-100 flex flex-col">
                <div className="p-4 border-b border-brand-100">
                    <button
                        onClick={createChat}
                        className="w-full flex items-center justify-center gap-2 bg-brand-900 text-white px-4 py-2.5 rounded-full font-medium text-sm hover:bg-brand-800 transition-all shadow"
                    >
                        <Plus className="w-4 h-4" />
                        New Chat
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto p-2 space-y-1">
                    {chats.map((c) => (
                        <ChatListItem
                            key={c.id}
                            chat={c}
                            active={c.id === activeId}
                            onSelect={() => setActiveId(c.id)}
                            onDelete={() => deleteChat(c.id)}
                        />
                    ))}
                </div>

                {/* Sidebar routing panel — latest assistant message's gating info */}
                {latestRouting && (
                    <div className="border-t border-brand-100 p-4 space-y-3 bg-brand-50/40">
                        <div className="flex items-center justify-between">
                            <span className="text-[10px] font-bold uppercase tracking-wider text-brand-500">
                                Last Route
                            </span>
                            {latestRouting.routingMode && (
                                <span className={`inline-flex items-center gap-1 text-[10px] uppercase tracking-wider font-bold px-2 py-0.5 rounded ${latestRouting.routingMode === 'hard' ? 'bg-amber-100 text-amber-700' :
                                    latestRouting.routingMode === 'blend' ? 'bg-purple-100 text-purple-700' :
                                        'bg-gray-100 text-gray-600'
                                    }`}>
                                    {latestRouting.routingMode}
                                </span>
                            )}
                        </div>

                        {/* Adapter chosen */}
                        <div>
                            {latestRouting.adapter === 'base_model' ? (
                                <span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full border bg-gray-100 text-gray-800 border-gray-200">
                                    <span className="w-1.5 h-1.5 rounded-full bg-current opacity-60" />
                                    base model
                                </span>
                            ) : (
                                <AdapterBadge domain={latestRouting.adapter} />
                            )}
                        </div>

                        {/* Compact gating bars */}
                        {latestRouting.gatingScores && (
                            <div className="space-y-1.5">
                                {Object.entries(latestRouting.gatingScores).map(([domain, score]) => (
                                    <SidebarGatingBar
                                        key={domain}
                                        domain={domain}
                                        score={score}
                                        isWinner={domain === latestRouting.adapter}
                                    />
                                ))}
                            </div>
                        )}

                        {latestRouting.confidence != null && (
                            <div className="text-[10px] font-mono text-brand-400 text-right">
                                conf: {(latestRouting.confidence * 100).toFixed(2)}%
                            </div>
                        )}
                    </div>
                )}
            </aside>

            {/* Main chat area */}
            <main className="flex-1 flex flex-col min-w-0">
                <header className="px-6 py-4 border-b border-brand-100 bg-white/80 backdrop-blur-sm flex items-center justify-between">
                    <div>
                        <h1 className="font-serif text-xl font-bold text-brand-900">
                            {activeChat?.title || 'Chat'}
                        </h1>
                        <p className="text-xs text-brand-500">AdaptRoute · Gemma-3-1B-It + dynamic LoRA routing</p>
                    </div>
                    {canRegenerate && (
                        <button
                            onClick={regenerate}
                            className="flex items-center gap-2 text-sm text-brand-700 hover:text-brand-900 px-3 py-1.5 rounded-full border border-brand-200 hover:bg-brand-50 transition"
                        >
                            <RefreshCw className="w-3.5 h-3.5" />
                            Regenerate
                        </button>
                    )}
                </header>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto px-6 py-8">
                    <div className="max-w-3xl mx-auto space-y-6">
                        {activeChat?.messages.length === 0 && <EmptyState onPick={setInput} />}

                        {activeChat?.messages.map((m, i) => (
                            <Message key={i} message={m} />
                        ))}

                        {error && (
                            <div className="bg-red-50 border border-red-200 rounded-2xl p-4 text-sm text-red-800 flex gap-3">
                                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                                <div>{error}</div>
                            </div>
                        )}

                        <div ref={messagesEndRef} />
                    </div>
                </div>

                {/* Input */}
                <div className="border-t border-brand-100 bg-white/80 backdrop-blur-sm px-6 py-4">
                    <div className="max-w-3xl mx-auto">
                        <div className="relative flex items-end gap-2 p-3 rounded-2xl border border-brand-200 bg-white shadow-sm focus-within:border-brand-500 transition-colors">
                            <textarea
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={onKeyDown}
                                placeholder="Ask AdaptRoute anything…"
                                rows={1}
                                disabled={loading}
                                className="flex-1 bg-transparent outline-none resize-none text-sm text-brand-900 placeholder-brand-400 max-h-40 disabled:opacity-60"
                                style={{ minHeight: '24px' }}
                            />
                            <button
                                onClick={sendMessage}
                                disabled={!input.trim() || workerStatus !== 'online' || loading}
                                className="flex items-center justify-center w-9 h-9 rounded-full bg-brand-900 text-white hover:bg-brand-800 transition disabled:opacity-40 disabled:cursor-not-allowed"
                                title="Send (Enter)"
                            >
                                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                            </button>
                        </div>
                        <div className="text-xs text-brand-400 mt-2 text-center">
                            Enter to send · Shift+Enter for newline
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};

// ── Sidebar item ────────────────────────────────────────────────────────────
const ChatListItem = ({ chat, active, onSelect, onDelete }) => (
    <div
        onClick={onSelect}
        className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition ${active ? 'bg-brand-100 text-brand-900' : 'text-brand-700 hover:bg-brand-50'
            }`}
    >
        <MessageSquare className="w-4 h-4 shrink-0 opacity-60" />
        <span className="flex-1 text-sm truncate">{chat.title}</span>
        <button
            onClick={(e) => { e.stopPropagation(); onDelete(); }}
            className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 hover:text-red-600 rounded transition"
            title="Delete"
        >
            <Trash2 className="w-3.5 h-3.5" />
        </button>
    </div>
);

// ── Empty state ─────────────────────────────────────────────────────────────
const EmptyState = ({ onPick }) => {
    return (
        <div className="text-center py-16">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-brand-100 mb-6">
                <Sparkles className="w-7 h-7 text-brand-700" />
            </div>
            <h2 className="font-serif text-3xl font-bold text-brand-900 mb-3">How can I help today?</h2>
            <p className="text-brand-600 mb-8">Watch the gating network pick the right expert in real time.</p>
        </div>
    );
};

// ── Message bubble ──────────────────────────────────────────────────────────
const Message = ({ message }) => {
    const isUser = message.role === 'user';
    const [copied, setCopied] = useState(false);

    const copy = () => {
        navigator.clipboard.writeText(message.content || '').then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
        });
    };
    return (
        <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}
        >
            {!isUser && (
                <div className="shrink-0 w-8 h-8 rounded-full bg-brand-900 text-white flex items-center justify-center">
                    <Cpu className="w-4 h-4" />
                </div>
            )}
            <div className={`max-w-[75%] ${isUser ? 'order-1' : ''}`}>
                {/* Blocked indicator stays above bubble */}
                {message.blocked && (
                    <div className="mb-3">
                        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-100 border border-red-300">
                            <ShieldAlert className="w-4 h-4 text-red-600" />
                            <span className="text-xs font-semibold text-red-700">Blocked by Firewall</span>
                        </div>
                    </div>
                )}

                {/* Bubble — now with markdown */}
                <div
                    className={`rounded-2xl px-5 py-3 text-sm leading-relaxed ${isUser
                        ? 'bg-brand-900 text-white whitespace-pre-wrap'
                        : message.blocked
                            ? 'bg-red-50 border border-red-200 text-red-900 whitespace-pre-wrap'
                            : message.error
                                ? 'bg-yellow-50 border border-yellow-200 text-yellow-900 whitespace-pre-wrap'
                                : 'bg-white border border-brand-100 text-brand-900 shadow-sm'
                        }`}
                >
                    {message.blocked && <ShieldAlert className="w-4 h-4 inline mr-2 -mt-0.5" />}
                    {message.content ? (
                        <>
                            {isUser || message.blocked || message.error ? (
                                message.content
                            ) : (
                                <div className="markdown-body">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                        {message.content}
                                    </ReactMarkdown>
                                </div>
                            )}
                            {message.streaming && (
                                <span className="inline-block w-1.5 h-4 bg-brand-400 ml-0.5 align-middle animate-pulse" />
                            )}
                        </>
                    ) : (
                        <span className="inline-flex items-center gap-2 text-brand-400">
                            <Loader2 className="w-3 h-3 animate-spin" /> generating…
                        </span>
                    )}
                </div>

                {/* Footer: copy button + timing */}
                {!isUser && message.content && !message.streaming && (
                    <div className="flex items-center gap-3 mt-2 text-xs text-brand-500">
                        <button
                            onClick={copy}
                            className="flex items-center gap-1 hover:text-brand-800 transition"
                        >
                            {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                            {copied ? 'Copied' : 'Copy'}
                        </button>
                        {message.seconds != null && (
                            <span className="opacity-60">{message.seconds}s</span>
                        )}
                    </div>
                )}

                {/* ROUTING BADGES — moved below the bubble */}
                {!isUser && !message.blocked && message.adapter && (
                    <div className="mt-3 flex flex-wrap gap-2 items-center">
                        {message.routingMode && (
                            <span className={`inline-flex items-center gap-1 text-[10px] uppercase tracking-wider font-bold px-2 py-0.5 rounded ${message.routingMode === 'hard' ? 'bg-amber-100 text-amber-700' :
                                message.routingMode === 'blend' ? 'bg-purple-100 text-purple-700' :
                                    'bg-gray-100 text-gray-600'
                                }`}>
                                {message.routingMode} route
                            </span>
                        )}
                        {message.confidence != null && (
                            <span className="text-[10px] font-mono text-brand-400">
                                conf: {(message.confidence * 100).toFixed(3)}%
                            </span>
                        )}
                        <div className="w-full h-0" />
                        {message.adapter === 'base_model' ? (
                            <span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full border bg-gray-100 text-gray-800 border-gray-200">
                                <span className="w-1.5 h-1.5 rounded-full bg-current opacity-60" />
                                base model chosen
                            </span>
                        ) : (
                            <AdapterBadge domain={message.adapter} />
                        )}
                    </div>
                )}
            </div>
            {isUser && (
                <div className="shrink-0 w-8 h-8 rounded-full bg-brand-200 text-brand-800 flex items-center justify-center">
                    <User className="w-4 h-4" />
                </div>
            )}
        </motion.div>
    );

}

// ── Badges ──────────────────────────────────────────────────────────────────
const AdapterBadge = ({ domain }) => {
    const colors = {
        code: 'bg-blue-100 text-blue-800 border-blue-200',
        math: 'bg-purple-100 text-purple-800 border-purple-200',
        qa: 'bg-green-100 text-green-800 border-green-200',
        medical: 'bg-rose-100 text-rose-800 border-rose-200',
    };
    const cls = colors[domain] || 'bg-brand-100 text-brand-800 border-brand-200';
    return (
        <span className={`inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full border ${cls}`}>
            <span className="w-1.5 h-1.5 rounded-full bg-current opacity-60" />
            {domain} adapter
        </span>
    );
};

const GatingScoreBar = ({ domain, score, isWinner }) => {
    const colors = {
        code: 'bg-blue-400',
        math: 'bg-purple-400',
        qa: 'bg-green-400',
        medical: 'bg-rose-400',
    };
    const barColor = colors[domain] || 'bg-brand-400';
    const barWidth = `${Math.round(score * 100)}%`;

    return (
        <div className="flex items-center justify-between text-sm font-medium">
            <div className="w-20 text-brand-700 flex items-center gap-1.5">
                {isWinner && <span className="text-yellow-500 text-lg">★</span>}
                <span className="font-semibold capitalize">{domain}</span>
            </div>
            <div className="flex-1 mx-4 h-8 bg-brand-100 rounded-full overflow-hidden">
                <div
                    className={`h-full ${barColor} transition-all duration-500 flex items-center justify-center`}
                    style={{ width: barWidth }}
                >
                    {score > 0.15 && <span className="text-white text-xs font-bold">{(score * 100).toFixed(2)}%</span>}
                </div>
            </div>
            <div className="w-16 text-right text-brand-700 font-semibold">{(score * 100).toFixed(2)}</div>
        </div>
    );
};

const SidebarGatingBar = ({ domain, score, isWinner }) => {
    const colors = {
        code: 'bg-blue-400',
        math: 'bg-purple-400',
        qa: 'bg-green-400',
        medical: 'bg-rose-400',
    };
    const barColor = colors[domain] || 'bg-brand-400';
    const barWidth = `${Math.max(2, Math.round(score * 100))}%`;

    return (
        <div className="flex items-center gap-2 text-[11px]">
            <div className="w-14 text-brand-700 flex items-center gap-1">
                {isWinner && <span className="text-yellow-500">★</span>}
                <span className="font-semibold capitalize truncate">{domain}</span>
            </div>
            <div className="flex-1 h-1.5 bg-brand-100 rounded-full overflow-hidden">
                <div
                    className={`h-full ${barColor} transition-all duration-500`}
                    style={{ width: barWidth }}
                />
            </div>
            <div className="w-10 text-right text-brand-600 font-mono tabular-nums text-[10px]">
                {(score * 100).toFixed(1)}%
            </div>
        </div>
    );
};

const GatingBadge = ({ scores, winner }) => {
    if (!scores) return null;

    return (
        <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 p-5 bg-gradient-to-br from-brand-50 to-brand-50/50 border border-brand-100 rounded-xl space-y-4"
        >
            <div className="text-sm font-bold text-brand-700 uppercase tracking-wide">Gating Distribution</div>
            {Object.entries(scores).map(([domain, score]) => (
                <GatingScoreBar
                    key={domain}
                    domain={domain}
                    score={score}
                    isWinner={domain === winner}
                />
            ))}
        </motion.div>
    );
};

export default Demo;