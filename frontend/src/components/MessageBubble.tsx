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

        {/* Sources toggle */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div>
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors mt-1 ml-1"
            >
              {showSources ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              참고 출처 ({message.sources.length}건)
            </button>

            {showSources && (
              <div className="mt-2 space-y-2">
                {message.sources.map((src, i) => (
                  <div key={i} className="bg-source border border-source-border rounded-lg px-3 py-2.5 text-xs">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="font-medium text-muted-foreground">출처 #{i + 1}</span>
                      <span className="text-source-score font-semibold">
                        유사도: {(src.score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-foreground/80 leading-relaxed line-clamp-4 whitespace-pre-wrap">{src.text}</p>
                    {src.filename && (
                      <p className="mt-1.5 text-muted-foreground">📄 {src.filename}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
