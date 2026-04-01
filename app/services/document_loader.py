from pathlib import Path
from typing import List, Tuple
import pypdf
import pdfplumber
from app.utils.logger import logger

class DocumentLoader:
    #문서 파싱 서비스

    @staticmethod
    def load_pdf(file_path: str) -> Tuple[str, List[int]]:
        """PDF 파일을 텍스트로 변환하고 페이지별로 구분하여 반환"""
        try:
            text = ""
            page_numbers = []

            with pdfplumber.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                    page_numbers.append([page_idx] * len(page_text.splitl()))

            if not text.strip():
                logger.warning(f"PDF에서 텍스트를 추출할 수 없습니다.:{file_path}")

            logger.info(f"PDF 파싱 완료: {file_path}, 크기: {len(text)} 글자")
            return text, page_numbers
        except Exception as e:
            logger.error(f"PDF 파싱 오류: {file_path}, {str(e)}")
            raise

    
@staticmethod
def load_document(file_path: str) -> Tuple[str, List[int]]:
    """파일 확장자에 따라 적절한 로더를 선택하여 문서를 로드"""
    file_ext = Path(file_path).suffix.lower()
    if file_ext == ".pdf":
        return DocumentLoader.load_pdf(file_path)
    elif file_ext in [".txt", ".md"]:
        return DocumentLoader.load_text(file_path)
    else:
        raise ValueError(f"지원되지 않는 파일 형식: {file_ext}")