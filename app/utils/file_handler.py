import os
import uuid
from datetime import datetime


def generate_document_id() -> str:
    """
    타임스탬프 + UUID 앞 8자리를 조합한 문서 ID를 생성한다.
    타임스탬프 단독은 동일 초 충돌 위험, UUID 단독은 가독성 부재 → 두 방식을 결합한다.

    Returns:
        "doc_YYYY-MM-DD_HH-MM-SS_xxxxxxxx" 형식의 문자열
    """
    return f"doc_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{str(uuid.uuid4())[:8]}"


def sanitize_filename(filename: str) -> str:
    """
    업로드 파일명에서 경로 조작 시퀀스를 제거하고 순수 파일명만 반환한다.
    file.filename은 클라이언트가 완전히 제어하는 값이라, "../"나 절대 경로가 섞여 있으면
    upload_dir 밖에 파일을 쓰는 경로 순회(path traversal) 취약점으로 이어질 수 있다.

    Parameters:
        filename: 클라이언트가 전송한 원본 파일명

    Returns:
        디렉토리 구분자가 제거된 순수 파일명

    Raises:
        ValueError: 파일명이 없거나(None), 비어 있거나, "."/".." 뿐인 경우
    """
    if not filename:
        # multipart 요청에 파일명이 아예 없는 경우 file.filename이 None일 수 있다.
        raise ValueError("유효하지 않은 파일명입니다")
    # Windows 경로 구분자(\)까지 제거하기 위해 POSIX 구분자로 통일 후 basename만 취한다.
    name = os.path.basename(filename.replace("\\", "/"))
    if not name or name in (".", ".."):
        raise ValueError("유효하지 않은 파일명입니다")
    return name
