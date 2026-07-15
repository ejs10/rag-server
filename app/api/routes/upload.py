from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
from app.core.config import settings
from app.services.rag_pipeline import shared_rag_pipeline
from app.models.schemas import UploadResponse
from app.utils.file_handler import sanitize_filename
from app.utils.logger import logger

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    문서를 업로드하고 RAG 파이프라인으로 처리한다.
    파일 검증 → 디스크 저장 → 텍스트 파싱 → 청킹 → 임베딩 → 벡터 DB 저장 순서로 진행된다.

    Parameters:
        file: 업로드할 파일 (PDF, TXT, MD 허용)
    """
    try:
        # file.filename은 클라이언트가 제어하는 값이므로 경로 조작 시퀀스를 먼저 제거한다.
        try:
            safe_filename = sanitize_filename(file.filename)
        except ValueError:
            raise HTTPException(status_code=400, detail="유효하지 않은 파일명입니다.")

        # 허용 확장자 화이트리스트 방식. 블랙리스트보다 안전하다.
        allowed_extensions = [".pdf", ".txt", ".md"]
        file_ext = Path(safe_filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"지원되지 않는 파일 형식입니다. {allowed_extensions} 중 하나의 형식을 사용해주세요."
            )

        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / safe_filename

        contents = await file.read()

        # 파일 내용을 읽은 후에 크기를 확인한다 (읽기 전엔 정확한 크기를 모름).
        max_size = settings.MAX_UPLOAD_SIZE_MB
        file_size_mb = len(contents) / (1024 * 1024)
        if file_size_mb > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"파일 크기초과 :{file_size_mb:.2f} MB, {max_size} MB."
            )

        with open(file_path, "wb") as f:
            f.write(contents)

        # process_document()는 동기 함수이므로 run_in_threadpool로 감싸 이벤트 루프 블로킹을 방지한다.
        result = await run_in_threadpool(
            shared_rag_pipeline.process_document,
            file_path=str(file_path),
            filename=safe_filename
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
