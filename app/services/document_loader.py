from pathlib import Path
from typing import List, Tuple
import pdfplumber
from app.utils.logger import logger

class DocumentLoader:
    """문서 파싱 서비스"""
    
    @staticmethod
    def load_pdf(file_path: str) -> Tuple[str, List[int]]:
        """
        PDF 파일에서 텍스트 추출
        """
        try:
            text = ""
            page_numbers = []
            
            with pdfplumber.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages, 1):
                    # 페이지 텍스트 추출 (없으면 빈 문자열)
                    page_text = page.extract_text()
                    if not page_text:
                        page_text = ""
                        
                    text += page_text + "\n"
                    
                    # 띄어쓰기(단어) 기준으로 분할하여 페이지 번호 매핑
                    # (여기에 splitl 이라고 오타가 났던 것을 split으로 고침)
                    words = page_text.split()
                    page_numbers.extend([page_idx] * len(words))
            
            if not text.strip():
                logger.warning(f"PDF에서 텍스트를 추출할 수 없습니다: {file_path}")
            
            logger.info(f"PDF 파싱 완료: {file_path}, 크기: {len(text)} 글자")
            return text, page_numbers
            
        except Exception as e:
            logger.error(f"PDF 파싱 오류: {file_path}, {str(e)}")
            raise
    
    @staticmethod
    def load_text(file_path: str) -> Tuple[str, List[int]]:
        """
        텍스트 파일 로드
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 텍스트 파일은 전체를 1페이지로 간주
            words = text.split()
            page_numbers = [1] * len(words)
            
            logger.info(f"텍스트 파일 로드 완료: {file_path}, 크기: {len(text)} 글자")
            return text, page_numbers
            
        except Exception as e:
            logger.error(f"텍스트 파일 로드 오류: {file_path}, {str(e)}")
            raise
    
    @staticmethod
    def load_document(file_path: str) -> Tuple[str, List[int]]:
        """
        파일 확장자에 따라 적절한 로더(PDF 또는 TXT) 선택
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == ".pdf":
            return DocumentLoader.load_pdf(file_path)
        elif file_ext in [".txt", ".md"]:
            return DocumentLoader.load_text(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_ext}")