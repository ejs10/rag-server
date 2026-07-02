from pathlib import Path
from typing import List, Tuple
import pdfplumber
from app.utils.logger import logger


class DocumentLoader:
    """파일 형식에 따라 텍스트와 페이지 번호를 추출하는 문서 파싱 서비스."""

    @staticmethod
    def load_pdf(file_path: str) -> Tuple[str, List[int]]:
        """
        PDF 파일에서 텍스트를 추출한다. 스캔된 PDF(이미지 기반)는 추출 불가.

        Parameters:
            file_path: PDF 파일 경로

        Returns:
            (전체 텍스트, 단어별 페이지 번호 목록) 튜플.
            페이지 번호 목록은 각 단어가 속한 페이지를 나타내며, 청크 생성 시 출처 표시에 사용된다.
        """
        try:
            text = ""
            page_numbers = []

            with pdfplumber.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if not page_text:
                        page_text = ""

                    text += page_text + "\n"

                    words = page_text.split()
                    page_numbers.extend([page_idx] * len(words))

            if not text.strip():
                logger.warning(f"PDF에서 텍스트를 추출할 수 없습니다: {file_path}")

            logger.debug(f"PDF 파싱 완료: {file_path}, 크기: {len(text)} 글자")
            return text, page_numbers

        except Exception as e:
            logger.error(f"PDF 파싱 오류: {file_path}, {str(e)}")
            raise

    @staticmethod
    def load_text(file_path: str) -> Tuple[str, List[int]]:
        """
        텍스트(.txt, .md) 파일을 로드한다. 페이지 개념이 없으므로 전체를 1페이지로 처리한다.

        Parameters:
            file_path: 텍스트 파일 경로

        Returns:
            (전체 텍스트, 단어별 페이지 번호 목록) 튜플
        """
        try:
            # utf-8 명시: Windows 환경에서 기본값(cp949)으로 열면 한글이 깨질 수 있다.
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            words = text.split()
            page_numbers = [1] * len(words)

            logger.debug(f"텍스트 파일 로드 완료: {file_path}, 크기: {len(text)} 글자")
            return text, page_numbers

        except Exception as e:
            logger.error(f"텍스트 파일 로드 오류: {file_path}, {str(e)}")
            raise

    @staticmethod
    def load_document(file_path: str) -> Tuple[str, List[int]]:
        """
        파일 확장자에 따라 적절한 로더를 선택하는 통합 진입점.

        Parameters:
            file_path: 로드할 파일 경로

        Returns:
            (전체 텍스트, 단어별 페이지 번호 목록) 튜플
        """
        file_ext = Path(file_path).suffix.lower()

        if file_ext == ".pdf":
            return DocumentLoader.load_pdf(file_path)
        elif file_ext in [".txt", ".md"]:
            return DocumentLoader.load_text(file_path)
        else:
            # upload.py에서 이미 확장자 검증을 하지만, 직접 호출 경로를 대비해 여기서도 방어한다.
            raise ValueError(f"지원하지 않는 파일 형식: {file_ext}")
