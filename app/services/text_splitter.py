import re
from typing import List, Dict


class TextSplitter:
    """텍스트 청킹 서비스. 코드블록(```)은 분할하지 않고, 일반 텍스트는 글자 수 기준으로 분할한다."""

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128):
        """
        Parameters:
            chunk_size   : 청크 하나의 최대 글자 수
            chunk_overlap: 이전 청크와 겹치는 글자 수.
                           청크 경계에서 잘린 맥락을 앞뒤 청크 모두에 포함시켜 검색 품질을 높인다.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # re.DOTALL: 코드블록은 여러 줄이므로 .이 \n도 매칭하도록 설정
        self._code_block_pattern = re.compile(r'```(\w*)\n?(.*?)```', re.DOTALL)

    def split_text(self, text: str, page_numbers: List[int] = None) -> List[Dict]:
        """
        텍스트를 코드블록과 일반 텍스트로 분리한 뒤 각각 처리한다.
        코드블록은 chunk_size를 초과해도 분할하지 않는다.

        Parameters:
            text        : 분할할 전체 텍스트
            page_numbers: 단어별 페이지 번호 목록 (DocumentLoader에서 생성)

        Returns:
            {"text", "page", "is_code", "language"} 키를 가진 청크 딕셔너리 목록
        """
        if not text.strip():
            return []

        chunks = []
        segments = self._parse_segments(text)

        for segment in segments:
            if segment['is_code']:
                # 코드를 중간에 자르면 문법 오류가 있는 불완전한 코드가 되므로 항상 하나의 청크로 유지한다.
                chunk_text = segment['text'].strip()
                if chunk_text:
                    page = self._get_page_number(text, segment['start'], page_numbers)
                    chunks.append({
                        'text': chunk_text,
                        'page': page,
                        'is_code': True,
                        'language': segment.get('language', ''),
                    })
            else:
                text_chunks = self._split_text_segment(
                    segment_text=segment['text'],
                    segment_start=segment['start'],
                    full_text=text,
                    page_numbers=page_numbers,
                )
                chunks.extend(text_chunks)

        return chunks

    def _parse_segments(self, text: str) -> List[Dict]:
        """
        텍스트를 코드블록과 일반 텍스트 구간 목록으로 분리한다.

        Parameters:
            text: 파싱할 전체 텍스트

        Returns:
            {"text", "start", "is_code", "language"} 키를 가진 구간 딕셔너리 목록
        """
        segments = []
        last_end = 0

        for match in self._code_block_pattern.finditer(text):
            block_start, block_end = match.span()

            if block_start > last_end:
                segments.append({
                    'text': text[last_end:block_start],
                    'start': last_end,
                    'is_code': False,
                    'language': '',
                })

            segments.append({
                'text': match.group(0),
                'start': block_start,
                'is_code': True,
                'language': match.group(1),
            })

            last_end = block_end

        if last_end < len(text):
            segments.append({
                'text': text[last_end:],
                'start': last_end,
                'is_code': False,
                'language': '',
            })

        return segments

    def _split_text_segment(
        self,
        segment_text: str,
        segment_start: int,
        full_text: str,
        page_numbers: List[int],
    ) -> List[Dict]:
        """
        일반 텍스트 구간을 글자 수 기준으로 분할한다.

        Parameters:
            segment_text : 분할할 텍스트 구간
            segment_start: 전체 텍스트 내 이 구간의 시작 위치 (페이지 번호 계산용)
            full_text    : 원본 전체 텍스트
            page_numbers : 단어별 페이지 번호 목록

        Returns:
            청크 딕셔너리 목록
        """
        chunks = []
        start_idx = 0

        while start_idx < len(segment_text):
            end_idx = min(start_idx + self.chunk_size, len(segment_text))

            if end_idx < len(segment_text):
                # 단어 중간에서 자르지 않기 위해 줄바꿈 또는 공백 위치를 찾는다.
                last_newline = segment_text.rfind('\n', start_idx, end_idx)
                if last_newline > start_idx:
                    end_idx = last_newline
                else:
                    last_space = segment_text.rfind(' ', start_idx, end_idx)
                    if last_space > start_idx:
                        end_idx = last_space

            chunk_text = segment_text[start_idx:end_idx].strip()

            if chunk_text:
                page = self._get_page_number(
                    full_text,
                    segment_start + start_idx,
                    page_numbers,
                )
                chunks.append({
                    'text': chunk_text,
                    'page': page,
                    'is_code': False,
                    'language': '',
                })

            start_idx = end_idx - self.chunk_overlap if end_idx < len(segment_text) else len(segment_text)

        return chunks

    def _get_page_number(self, full_text: str, char_pos: int, page_numbers: List[int]) -> int:
        """
        텍스트 내 문자 위치를 단어 인덱스로 근사 변환해 페이지 번호를 반환한다.
        줄바꿈 수를 단어 위치의 근사값으로 사용한다.

        Parameters:
            full_text   : 원본 전체 텍스트
            char_pos    : 페이지를 조회할 문자 위치
            page_numbers: 단어별 페이지 번호 목록

        Returns:
            해당 위치의 추정 페이지 번호
        """
        if not page_numbers:
            return 1

        # page_numbers는 단어 하나당 하나의 항목을 가지므로(DocumentLoader 참고),
        # 같은 기준(공백 분리 단어 수)으로 char_pos 이전 단어 수를 세어야 인덱스가 맞는다.
        word_count = len(full_text[:char_pos].split())

        if word_count < len(page_numbers):
            return page_numbers[word_count]

        return page_numbers[-1] if page_numbers else 1
