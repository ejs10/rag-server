from pydantic import BaseModel
from typing import List, Optional


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    file_size: int
    upload_time: str
    total_chunks: int


class ChunkInfo(BaseModel):
    document_id: str
    chunk_index: int
    text: str
    page: Optional[int] = None  # PDF만 페이지 번호를 가짐. 텍스트 파일은 None.
    score: float


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str
    total_chunks: int


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    file_size: int
    upload_time: str
    total_chunks: int


class DocumentListResponse(BaseModel):
    count: int
    documents: List[DocumentResponse]


class QueryRequest(BaseModel):
    question: str
    document_id: Optional[str] = None  # None이면 전체 문서 검색
    session_id: str
    top_k: int = 4


class Source(BaseModel):
    document_id: str
    chunk_index: int
    score: float
    text: str  # 응답 크기 절약을 위해 앞 200자만 반환
    page: Optional[int] = None
    filename: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: dict


class LangGraphQueryRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    session_id: str
    top_k: int = 4
    use_rerank: bool = False
    use_query_rewriting: bool = True


class LangGraphQueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str
    rewritten_query: Optional[str] = None  # 재작성된 질문 (디버깅·투명성용)
    route: Optional[str] = None            # 사용된 검색 방식 ("vector"/"graph"/"hybrid")
    node_trace: Optional[List[str]] = None # 실행된 LangGraph 노드 순서 (디버깅용)


class DatasetExample(BaseModel):
    question: str
    expected_answer: Optional[str] = None
    context: Optional[str] = None
    document_id: Optional[str] = None


class CreateDatasetRequest(BaseModel):
    dataset_name: str
    description: str = ""
    examples: List[DatasetExample]


class CreateDatasetResponse(BaseModel):
    dataset_id: Optional[str]
    dataset_name: str
    example_count: int
    status: str


class RunEvalRequest(BaseModel):
    dataset_name: str
    experiment_prefix: str = "rag-eval"


class RunEvalResponse(BaseModel):
    status: str
    experiment_prefix: str
    dataset_name: str
    error: Optional[str] = None


class FeedbackRequest(BaseModel):
    run_id: str
    key: str
    score: float
    comment: str = ""
