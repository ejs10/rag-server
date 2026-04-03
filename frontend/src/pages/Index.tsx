import { useState, useEffect, useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import DocumentSidebar from "@/components/DocumentSidebar";
import ChatArea from "@/components/ChatArea";
import { fetchDocuments, type Document } from "@/lib/api";

const Index = () => {
  const [sessionId] = useState(() => uuidv4());
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);

  const loadDocs = useCallback(async () => {
    try {
      const data = await fetchDocuments();
      setDocuments(data.documents);
    } catch {
      // silently fail
    }
  }, []);

  useEffect(() => {
    loadDocs();
  }, [loadDocs]);

  return (
    <div className="flex h-screen w-full overflow-hidden">
      <DocumentSidebar
        documents={documents}
        selectedDocId={selectedDocId}
        onSelectDoc={setSelectedDocId}
        onUploadSuccess={loadDocs}
      />
      <ChatArea sessionId={sessionId} selectedDocId={selectedDocId} />
    </div>
  );
};

export default Index;
