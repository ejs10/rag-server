import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { ChatSource } from "@/lib/api";

interface Message {
  role: "user" | "ai";
  content: string;
  sources?: ChatSource[];
}

export default function MessageBubble({ message }: { message: Message }) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === "user";

  return (
    <div className={`flex items-start gap-3 ${isUser ? "flex-row-reverse" : ""} max-w-3xl ${isUser ? "ml-auto" : ""}`}>
      {/* Avatar */}
      <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm flex-shrink-0 ${
        isUser ? "bg-primary text-primary-foreground" : "bg-muted"
      }`}>
        {isUser ? "👤" : "🤖"}
      </div>

      {/* Bubble */}
      <div className="flex flex-col gap-1 min-w-0 max-w-[85%]">
        <div className={`px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "bg-chat-user text-chat-user-foreground rounded-2xl rounded-tr-md"
            : "bg-chat-ai text-chat-ai-foreground rounded-2xl rounded-tl-md"
        }`}>
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose prose-sm max-w-none prose-p:my-1 prose-headings:my-2 prose-li:my-0.5 prose-pre:bg-foreground/5 prose-pre:rounded-lg">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* 출처 패널: URL 소스 기능 추가 시 false를 조건으로 교체 */}
        {false && !isUser && message.sources && message.sources.length > 0 && (() => {
          const uniqueSources = Array.from(
            new Map(message.sources.map((s) => [s.filename ?? s.document_id, s])).values()
          );
          return (
            <div>
              <button
                onClick={() => setShowSources(!showSources)}
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors mt-1 ml-1"
              >
                {showSources ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                참고 출처 ({uniqueSources.length}건)
              </button>

              {showSources && (
                <div className="mt-2 space-y-1">
                  {uniqueSources.map((src, i) => (
                    <div key={i} className="flex items-center gap-2 bg-source border border-source-border rounded-md px-3 py-1.5 text-xs text-muted-foreground">
                      <span className="text-base leading-none">📄</span>
                      <span className="font-medium text-foreground/75 truncate">
                        {src.filename ?? src.document_id ?? `출처 #${i + 1}`}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })()}
      </div>
    </div>
  );
}
