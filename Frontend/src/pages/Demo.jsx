import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Send, StopCircle, Plus, Trash2, MessageSquare, Copy, RefreshCw,
    Loader2, Cpu, ShieldAlert, AlertTriangle, User, Sparkles, Check,
} from 'lucide-react';
import { streamChat, checkHealth, getWorkerUrl } from '../lib/adaptroute';

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
    const [streaming, setStreaming] = useState(false);
    const [error, setError] = useState(null);
    const [workerStatus, setWorkerStatus] = useState('checking');
    const abortRef = useRef(null);
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

    // Persist chats
    useEffect(() => {
        if (chats.length > 0) saveChats(chats);
    }, [chats]);

    const activeChat = useMemo(
        () => chats.find((c) => c.id === activeId) || null,
        [chats, activeId]
    );

    // Auto-scroll to bottom on new message/token
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [activeChat?.messages]);

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

    // ── Send / stream ──
    const runStream = useCallback(async (historyMessages) => {
        if (workerStatus !== 'online') {
            setError('Worker is not reachable. Start the Colab cell and check VITE_WORKER_URL.');
            return;
        }
        setError(null);
        setStreaming(true);

        // Insert a placeholder assistant message
        const placeholder = {
            role: 'assistant',
            content: '',
            adapter: null,
            gatingScores: null,
            firewallLabel: null,
            streaming: true,
            seconds: null,
        };
        updateActiveChat((c) => ({ ...c, messages: [...c.messages, placeholder] }));

        const controller = new AbortController();
        abortRef.current = controller;

        try {
            await streamChat({
                messages: historyMessages.map(({ role, content }) => ({ role, content })),
                signal: controller.signal,
                onEvent: (event) => {
                    updateActiveChat((c) => {
                        const msgs = [...c.messages];
                        const last = msgs[msgs.length - 1];
                        if (!last || last.role !== 'assistant') return c;

                        if (event.type === 'meta') {
                            msgs[msgs.length - 1] = {
                                ...last,
                                adapter: event.adapter_used,
                                gatingScores: event.gating_scores,
                                firewallLabel: event.firewall_label,
                            };
                        } else if (event.type === 'token') {
                            msgs[msgs.length - 1] = { ...last, content: last.content + event.text };
                        } else if (event.type === 'done') {
                            msgs[msgs.length - 1] = { ...last, streaming: false, seconds: event.seconds };
                        } else if (event.type === 'blocked') {
                            msgs[msgs.length - 1] = {
                                ...last,
                                streaming: false,
                                blocked: true,
                                content: event.message || 'Query blocked by firewall.',
                                firewallLabel: event.firewall_label,
                            };
                        } else if (event.type === 'error') {
                            msgs[msgs.length - 1] = {
                                ...last,
                                streaming: false,
                                error: true,
                                content: event.message || 'Worker error.',
                            };
                        }
                        return { ...c, messages: msgs };
                    });
                },
            });
        } catch (e) {
            if (e.name === 'AbortError') {
                updateActiveChat((c) => {
                    const msgs = [...c.messages];
                    const last = msgs[msgs.length - 1];
                    if (last && last.role === 'assistant') {
                        msgs[msgs.length - 1] = {
                            ...last,
                            streaming: false,
                            content: last.content + '\n\n[stopped]',
                        };
                    }
                    return { ...c, messages: msgs };
                });
            } else {
                setError(e.message || String(e));
                updateActiveChat((c) => ({
                    ...c,
                    messages: c.messages.slice(0, -1), // drop placeholder on failure
                }));
            }
        } finally {
            setStreaming(false);
            abortRef.current = null;
        }
    }, [updateActiveChat, workerStatus]);

    const sendMessage = async () => {
        const text = input.trim();
        if (!text || streaming || !activeChat) return;

        const userMsg = { role: 'user', content: text };
        const isFirst = activeChat.messages.length === 0;
        updateActiveChat((c) => ({
            ...c,
            title: isFirst ? titleFromMessage(text) : c.title,
            messages: [...c.messages, userMsg],
        }));
        setInput('');

        const history = [...activeChat.messages, userMsg];
        await runStream(history);
    };

    const regenerate = async () => {
        if (!activeChat || streaming) return;
        // Drop the last assistant message, then rerun on history up to the last user message
        const msgs = [...activeChat.messages];
        while (msgs.length > 0 && msgs[msgs.length - 1].role === 'assistant') {
            msgs.pop();
        }
        if (msgs.length === 0) return;
        updateActiveChat((c) => ({ ...c, messages: msgs }));
        await runStream(msgs);
    };

    const stop = () => {
        abortRef.current?.abort();
    };

    const onKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const canRegenerate = activeChat?.messages.some((m) => m.role === 'assistant') && !streaming;

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
                <div className="p-4 border-t border-brand-100 text-xs text-brand-500 flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${workerStatus === 'online' ? 'bg-green-500 animate-pulse' :
                            workerStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500 animate-pulse'
                        }`} />
                    Worker {workerStatus}
                </div>
            </aside>

            {/* Main chat area */}
            <main className="flex-1 flex flex-col min-w-0">
                <header className="px-6 py-4 border-b border-brand-100 bg-white/80 backdrop-blur-sm flex items-center justify-between">
                    <div>
                        <h1 className="font-serif text-xl font-bold text-brand-900">
                            {activeChat?.title || 'Chat'}
                        </h1>
                        <p className="text-xs text-brand-500">AdaptRoute · Qwen2.5-1.5B + dynamic LoRA routing</p>
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
                                disabled={streaming || workerStatus !== 'online'}
                                className="flex-1 bg-transparent outline-none resize-none text-sm text-brand-900 placeholder-brand-400 max-h-40 disabled:opacity-60"
                                style={{ minHeight: '24px' }}
                            />
                            {streaming ? (
                                <button
                                    onClick={stop}
                                    className="flex items-center justify-center w-9 h-9 rounded-full bg-red-500 text-white hover:bg-red-600 transition"
                                    title="Stop"
                                >
                                    <StopCircle className="w-4 h-4" />
                                </button>
                            ) : (
                                <button
                                    onClick={sendMessage}
                                    disabled={!input.trim() || workerStatus !== 'online'}
                                    className="flex items-center justify-center w-9 h-9 rounded-full bg-brand-900 text-white hover:bg-brand-800 transition disabled:opacity-40 disabled:cursor-not-allowed"
                                    title="Send (Enter)"
                                >
                                    <Send className="w-4 h-4" />
                                </button>
                            )}
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
    const examples = [
        { label: 'Code', text: 'Write a Python function for binary search with comments.' },
        { label: 'Math', text: 'Find the derivative of f(x) = 3x^2 + sin(x).' },
        { label: 'QA', text: 'Who invented the World Wide Web and when?' },
        { label: 'Medical', text: 'What are the common early symptoms of type 2 diabetes?' },
    ];
    return (
        <div className="text-center py-16">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-brand-100 mb-6">
                <Sparkles className="w-7 h-7 text-brand-700" />
            </div>
            <h2 className="font-serif text-3xl font-bold text-brand-900 mb-3">How can I help today?</h2>
            <p className="text-brand-600 mb-8">Watch the gating network pick the right expert in real time.</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-xl mx-auto">
                {examples.map((ex) => (
                    <button
                        key={ex.label}
                        onClick={() => onPick(ex.text)}
                        className="text-left p-4 rounded-2xl border border-brand-200 bg-white hover:border-brand-400 hover:shadow-md transition"
                    >
                        <div className="text-xs font-semibold text-brand-500 uppercase tracking-wider mb-1">
                            {ex.label}
                        </div>
                        <div className="text-sm text-brand-800">{ex.text}</div>
                    </button>
                ))}
            </div>
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
                {/* Adapter badge */}
                {!isUser && message.adapter && (
                    <div className="flex items-center gap-2 mb-2">
                        <AdapterBadge domain={message.adapter} />
                        {message.gatingScores && (
                            <GatingBadge scores={message.gatingScores} winner={message.adapter} />
                        )}
                    </div>
                )}

                {/* Bubble */}
                <div
                    className={`rounded-2xl px-5 py-3 text-sm leading-relaxed whitespace-pre-wrap ${isUser
                            ? 'bg-brand-900 text-white'
                            : message.blocked
                                ? 'bg-red-50 border border-red-200 text-red-900'
                                : message.error
                                    ? 'bg-yellow-50 border border-yellow-200 text-yellow-900'
                                    : 'bg-white border border-brand-100 text-brand-900 shadow-sm'
                        }`}
                >
                    {message.blocked && <ShieldAlert className="w-4 h-4 inline mr-2 -mt-0.5" />}
                    {message.content || (message.streaming && (
                        <span className="inline-flex items-center gap-2 text-brand-400">
                            <Loader2 className="w-3 h-3 animate-spin" /> thinking…
                        </span>
                    ))}
                    {message.streaming && message.content && (
                        <span className="inline-block w-1.5 h-4 bg-brand-500 ml-0.5 animate-pulse align-middle" />
                    )}
                </div>

                {/* Footer actions */}
                {!isUser && !message.streaming && message.content && (
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
            </div>
            {isUser && (
                <div className="shrink-0 w-8 h-8 rounded-full bg-brand-200 text-brand-800 flex items-center justify-center">
                    <User className="w-4 h-4" />
                </div>
            )}
        </motion.div>
    );
};

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

const GatingBadge = ({ scores, winner }) => {
    const topScore = scores[Object.keys(scores).find((k) => k.includes(winner))] ?? 0;
    return (
        <span className="inline-flex items-center gap-1 text-xs text-brand-500 px-2 py-0.5 rounded-full bg-brand-50 border border-brand-100">
            {(topScore * 100).toFixed(1)}% confidence
        </span>
    );
};

export default Demo;