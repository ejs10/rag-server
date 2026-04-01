from typing import List, Dict

class TextSplitter:
    """텍스트 청킹 서비스"""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, page_numbers: List[int] = None) -> List[str]:
        """
        텍스트를 청크로 분할
        
        Returns:
            청크 리스트 (각 청크는 {text, page} 포함)
        """
        if not text.strip():
            return []
        
        chunks = []
        start_idx = 0

        while start_idx < len(text):
            end_idx = min(start_idx + self.chunk_size, len(text))

            if end_idx < len(text):
                last_space = text.rfind('\n', start_idx, end_idx)
                if last_space > start_idx:
                    end_idx = last_space
                else:
                    last_space = text.rfind(' ', start_idx, end_idx)
                    if last_space > start_idx:
                        end_idx = last_space

            chunk_text = text[start_idx:end_idx].strip()

            if chunk_text:
                page = 1
                if page_numbers:
                    word_count = 0
                    for i, char in enumerate(text[:start_idx]):
                        if char == '\n':
                            word_count += 1
                    if word_count < len(page_numbers):
                        page = page_numbers[word_count]  # 페이지 번호 할당

                chunks.append({'text': chunk_text, 'page': page})
            start_idx = end_idx - self.chunk_overlap if end_idx < len(text) else len(text)

        return chunks