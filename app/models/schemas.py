from pydantic import BaseModel
from typing import List, Optional

class DocumentMetadata(BaseModel):
    """문서 메타데이터"""
    document_id: str #문서 ID
    filename: str #파일명
    file_size: int #파일 크기 (바이트)
    upload_time: str #업로드 시간 (ISO 형식)
    total_chunks: int #총 청크 수

#청크정보 모델
class ChunkInfo(BaseModel):
    document_id: str #문서 ID
    chunk_index: int #청크 인덱스
    text: str #청크 텍스트
    page: Optional[int] = None #페이지 정보 (선택적)
    score: float #유사도 점수 (검색 결과에서 사용)


class UploadResponse(BaseModel):
    """파일 업로드 응답"""
    document_id: str #문서 ID
    filename: str #파일명
    status: str #상태
    message: str #메시지
    total_chunks: int #총 청크 수


class DocumentResponse(BaseModel):
    """문서 조회 응답"""
    document_id: str #문서 ID
    filename: str #파일명
    file_size: int #파일 크기 (바이트)
    upload_time: str #업로드 시간 (ISO 형식)
    total_chunks: int #총 청크 수


class DocumentListResponse(BaseModel):
    """문서 목록 응답"""
    count: int #문서 수
    documents: List[DocumentResponse] #문서 목록


class QueryRequest(BaseModel):
    """질문 요청"""
    question: str #질문 텍스트
    document_id: Optional[str] = None  # 특정 문서만 검색 시
    session_id: str #세션 ID (대화 세션 구분용)
    top_k: int = 4 #검색 결과로 반환할 상위 K개 청크 수


class Source(BaseModel):
    """검색 결과 출처"""
    document_id: str #문서 ID
    chunk_index: int #청크 인덱스
    score: float #유사도 점수
    text: str #청크 텍스트
    page: Optional[int] = None #페이지 정보 (선택적)


class QueryResponse(BaseModel):
    """질문 응답"""
    answer: str #LLM이 생성한 답변 텍스트
    sources: List[Source] #검색 결과 출처 목록
    session_id: str #세션 ID (대화 세션 구분용)


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str #상태 (예: "ok")0
    version: str #버전 정보
    environment: dict #환경 정보 (예: {"debug": true})