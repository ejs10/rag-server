import os
import uuid
from pathlib import Path
from datetime import datetime

# 고유한 문서 ID 생성 함수
def generate_document_id() -> str:
    return f"doc_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{str(uuid.uuid4())[:8]}"

def save_uploaded_file(file_path: str, upload_dir: str) -> str:
    # 업로드 디렉토리가 존재하지 않으면 생성
    Path(upload_dir).mkdir(parents=True, exist_ok=True)

    frile_name = os.path.basename(file_path)
    save_path = os.path.join(upload_dir, frile_name)

    # 파일을 업로드 디렉토리로 이동
    # destination = os.path.join(upload_dir, os.path.basename(file_path))
    # os.rename(file_path, destination)

    return save_path

def get_file_extension(file_path: str) -> str:
    return os.path.splitext(file_path)[1].lower()