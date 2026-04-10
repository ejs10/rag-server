from fastapi import APIRouter, HTTPException
from app.services.rag_pipeline import shared_rag_pipeline  # 공통 파이프라인 가져오기
from app.models.schemas import DocumentResponse, DocumentListResponse
from app.utils.logger import logger

router = APIRouter(prefix="/documents", tags=["documents"])

@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """업로드된 문서 리스트 조회"""
    try:
        documents = shared_rag_pipeline.get_documents_metadata()
        return DocumentListResponse(
            count=len(documents),
            documents=[
                DocumentResponse(
                    document_id=doc["document_id"],
                    filename=doc["filename"],
                    file_size=doc["file_size"],
                    upload_time=doc["upload_time"],
                    total_chunks=doc["total_chunks"]
                ) for doc in documents
            ]
        )
    except Exception as e:
        logger.error(f"문서 리스트 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="문서 리스트 조회 중 오류가 발생했습니다.")
    
@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """특정 문서 정보 조회"""
    try:
        doc = shared_rag_pipeline.documents_metadata.get(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
        return DocumentResponse(\
            document_id=doc["document_id"],
            filename=doc["filename"],
            file_size=doc["file_size"],
            upload_time=doc["upload_time"],
            total_chunks=doc["total_chunks"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"문서 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="문서 조회 중 오류가 발생했습니다.")