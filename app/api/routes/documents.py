from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.services.rag_pipeline import shared_rag_pipeline
from app.models.schemas import DocumentResponse, DocumentListResponse
from app.utils.logger import logger

# upload.py와 같은 prefix를 공유하며 HTTP 메서드(GET/DELETE)로 구분된다.
router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """
    업로드된 전체 문서 목록을 반환한다.
    DB 쿼리 없이 메모리의 metadata 딕셔너리에서 직접 읽으므로 빠르다.
    """
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
    """
    특정 문서의 메타데이터를 반환한다.

    Parameters:
        document_id: 경로 파라미터로 전달되는 문서 고유 ID
    """
    try:
        doc = shared_rag_pipeline.documents_metadata.get(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
        return DocumentResponse(
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


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    ChromaDB 벡터 청크와 메타데이터에서 문서를 삭제한다.

    Parameters:
        document_id: 삭제할 문서의 고유 ID
    """
    try:
        deleted = await run_in_threadpool(
            shared_rag_pipeline.delete_document, document_id
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
        return {"status": "success", "document_id": document_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"문서 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail="문서 삭제 중 오류가 발생했습니다.")
