import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { FileText, Upload, Loader2, Check, X } from "lucide-react";
import { uploadDocument, type Document } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  documents: Document[];
  selectedDocId: string | null;
  onSelectDoc: (id: string | null) => void;
  onUploadSuccess: () => void;
}

export default function DocumentSidebar({ documents, selectedDocId, onSelectDoc, onUploadSuccess }: Props) {
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "success" | "error">("idle");

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    setUploading(true);
    setUploadStatus("idle");
    try {
      for (const file of acceptedFiles) {
        await uploadDocument(file);
      }
      setUploadStatus("success");
      onUploadSuccess();
      setTimeout(() => setUploadStatus("idle"), 2000);
    } catch {
      setUploadStatus("error");
      setTimeout(() => setUploadStatus("idle"), 3000);
    } finally {
      setUploading(false);
    }
  }, [onUploadSuccess]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "text/plain": [".txt"],
    },
    disabled: uploading,
  });

  const getFileIcon = (filename: string) => {
    const ext = filename.split(".").pop()?.toLowerCase();
    if (ext === "pdf") return <FileText className="h-4 w-4 text-red-400" />;
    return <FileText className="h-4 w-4 text-blue-400" />;
  };

  return (
    <aside className="w-72 min-w-[18rem] flex flex-col bg-sidebar-panel text-sidebar-panel-foreground border-r border-sidebar-panel-border h-screen">
      {/* Header */}
      <div className="px-5 pt-6 pb-4">
        <h1 className="text-lg font-bold text-sidebar-panel-bright tracking-tight">📄 DocChat</h1>
        <p className="text-xs mt-1 opacity-60">문서 기반 AI Q&A</p>
      </div>

      {/* Upload zone */}
      <div className="px-4 pb-4">
        <div
          {...getRootProps()}
          className={cn(
            "border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all duration-200",
            isDragActive
              ? "border-sidebar-panel-accent bg-sidebar-panel-accent/10"
              : "border-sidebar-panel-border hover:border-sidebar-panel-foreground/40",
            uploading && "pointer-events-none opacity-60"
          )}
        >
          <input {...getInputProps()} />
          {uploading ? (
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="h-6 w-6 animate-spin text-sidebar-panel-accent" />
              <span className="text-xs">업로드 중...</span>
            </div>
          ) : uploadStatus === "success" ? (
            <div className="flex flex-col items-center gap-2 text-green-400">
              <Check className="h-6 w-6" />
              <span className="text-xs">업로드 완료!</span>
            </div>
          ) : uploadStatus === "error" ? (
            <div className="flex flex-col items-center gap-2 text-red-400">
              <X className="h-6 w-6" />
              <span className="text-xs">업로드 실패</span>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Upload className="h-6 w-6 opacity-50" />
              <span className="text-xs">PDF, TXT 파일을 드래그하거나<br />클릭하여 업로드</span>
            </div>
          )}
        </div>
      </div>

      {/* Document list */}
      <div className="px-4 pb-2">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold uppercase tracking-wider opacity-50">문서 목록</h2>
          {selectedDocId && (
            <button onClick={() => onSelectDoc(null)} className="text-[10px] text-sidebar-panel-accent hover:underline">
              선택 해제
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin px-3 pb-4">
        {documents.length === 0 ? (
          <p className="text-xs text-center opacity-40 mt-8">아직 업로드된 문서가 없습니다</p>
        ) : (
          <ul className="space-y-1">
            {documents.map((doc) => (
              <li key={doc.document_id}>
                <button
                  onClick={() => onSelectDoc(selectedDocId === doc.document_id ? null : doc.document_id)}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left text-sm transition-colors",
                    selectedDocId === doc.document_id
                      ? "bg-sidebar-panel-accent/20 text-sidebar-panel-bright"
                      : "hover:bg-sidebar-panel-hover text-sidebar-panel-foreground"
                  )}
                >
                  {getFileIcon(doc.filename)}
                  <span className="truncate flex-1">{doc.filename}</span>
                  {selectedDocId === doc.document_id && (
                    <span className="w-2 h-2 rounded-full bg-sidebar-panel-accent flex-shrink-0" />
                  )}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}
