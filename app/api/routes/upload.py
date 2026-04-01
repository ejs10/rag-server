from fastapi import APIRouter, File, UploadFile, HTTPException
from pathlib import Path
from app.core.config import settings
from app.services.rag_pipeline import RAGPipeline
from app.models.schemas import UploadResponse
from app.utils.logger import logger

router = APIRouter(prefix="/documents", tags=["documents"])
rag_pipeline = RAGPipeline()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    PDF/텍스트 문서 업로드
    
    - 파일 검증
    - 문서 처리 (파싱 -> 청킹 -> 임베딩)
    - 벡터 DB 저장
    """
    try:
        # 파일 형식 검증
        allowed_extensions = [".pdf", ".txt", ".md"]
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"지원되지 않는 파일 형식입니다. {allowed_extensions} 중 하나의 형식을 사용해주세요."
            )
        # 파일 크기 검증
        contents = await file.read()
        file_size = len(contents) / (1024 * 1024)  # MB 단위
        if file_size > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"파일 크기초과 :{file_size:.2f} MB, {settings.MAX_FILE_SIZE_MB} MB."
            )
        
        # 파일 저장
        file_path = Path(settings.UPLOAD_DIR) / file.filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(contents)

        # RAG 파이프라인 처리 및 벡터 DB 저장
        result = rag_pipeline.process_document(
            file_path=str(file_path),
            filename=file.filename
        )

        return UploadResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            status="success",
            message="문서가 성공적으로 처리되었습니다",
            total_chunks=result["total_chunks"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 업로드 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"문서 처리 중 오류가 발생했습니다: {str(e)}"
        )