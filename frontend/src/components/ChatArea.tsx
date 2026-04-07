import { useState, useRef, useEffect } from "react";
import { Send } from "lucide-react";
import { sendQuery, type ChatSource } from "@/lib/api";
import MessageBubble from "./MessageBubble";

interface Message {
  id: string;
  role: "user" | "ai";
  content: string;
  sources?: ChatSource[];
}

interface Props {
  sessionId: string;
  selectedDocId: string | null;
}

export default function ChatArea({ sessionId, selectedDocId }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const q = input.trim();
    if (!q || loading) return;

    const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: q };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await sendQuery(q, sessionId, selectedDocId);
      const aiMsg: Message = {
        id: crypto.randomUUID(),
        role: "ai",
        content: res.answer,
        sources: res.sources,
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch {
      const errMsg: Message = {
        id: crypto.randomUUID(),
        role: "ai",
        content: "죄송합니다, 답변 생성 중 오류가 발생했습니다. 다시 시도해 주세요.",
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex-1 flex flex-col h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border px-6 py-4 flex items-center gap-3">
        <h2 className="font-semibold text-foreground">AI 문서 Q&A</h2>
        {selectedDocId && (
          <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded-full font-medium">
            문서 필터 활성
          </span>
        )}
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 space-y-4">
        {messages.length === 0 && !loading && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md">
              <div className="text-5xl mb-4">💬</div>
              <h3 className="text-lg font-semibold text-foreground mb-2">문서에 대해 질문하세요</h3>
              <p className="text-sm text-muted-foreground">
                왼쪽에서 문서를 업로드한 후, 여기서 질문하면 AI가 문서 내용을 기반으로 답변합니다.
              </p>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {loading && (
          <div className="flex items-start gap-3 max-w-2xl">
            <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center text-sm flex-shrink-0">🤖</div>
            <div className="bg-chat-ai text-chat-ai-foreground px-4 py-3 rounded-2xl rounded-tl-md">
              <div className="flex items-center gap-1.5">
                <span className="text-sm text-muted-foreground">답변을 생성 중입니다</span>
                <span className="typing-dot w-1.5 h-1.5 rounded-full bg-muted-foreground inline-block" />
                <span className="typing-dot w-1.5 h-1.5 rounded-full bg-muted-foreground inline-block" />
                <span className="typing-dot w-1.5 h-1.5 rounded-full bg-muted-foreground inline-block" />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="border-t border-border px-6 py-4">
        <div className="flex items-end gap-3 max-w-3xl mx-auto">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="질문을 입력하세요..."
            rows={1}
            className="flex-1 resize-none rounded-xl border border-input bg-card px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring transition-shadow"
            style={{ maxHeight: 120 }}
            onInput={(e) => {
              const t = e.currentTarget;
              t.style.height = "auto";
              t.style.height = Math.min(t.scrollHeight, 120) + "px";
            }}
          />
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="h-11 w-11 rounded-xl bg-primary text-primary-foreground flex items-center justify-center hover:opacity-90 transition-opacity disabled:opacity-40 flex-shrink-0"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </form>
    </div>
  );
}
