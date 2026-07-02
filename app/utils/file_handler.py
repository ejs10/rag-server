import os
import uuid
from pathlib import Path
from datetime import datetime


def generate_document_id() -> str:
    """
    타임스탬프 + UUID 앞 8자리를 조합한 문서 ID를 생성한다.
    타임스탬프 단독은 동일 초 충돌 위험, UUID 단독은 가독성 부재 → 두 방식을 결합한다.

    Returns:
        "doc_YYYY-MM-DD_HH-MM-SS_xxxxxxxx" 형식의 문자열
    """
    return f"doc_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{str(uuid.uuid4())[:8]}"


def save_uploaded_file(file_path: str, upload_dir: str) -> str:
    """
    업로드 디렉토리에 파일 저장 경로를 반환한다.
    실제 파일 복사는 upload.py에서 직접 처리하므로 여기서는 경로 계산만 수행한다.

    Parameters:
        file_path : 원본 파일 경로
        upload_dir: 저장할 대상 디렉토리

    Returns:
        최종 저장 경로 문자열
    """
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    file_name = os.path.basename(file_path)
    save_path = os.path.join(upload_dir, file_name)
    return save_path


def get_file_extension(file_path: str) -> str:
    """
    파일 확장자를 소문자로 정규화해 반환한다.

    Returns:
        ".pdf", ".txt" 형식의 소문자 확장자 문자열
    """
    return os.path.splitext(file_path)[1].lower()
