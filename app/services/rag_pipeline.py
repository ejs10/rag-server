from typing import List, Dict, Optional
import os
from datetime import datetime
import json
from app.services.document_loader import DocumentLoader
from app.services.vector_store import VectorStore
from app.services.text_splitter import TextSplitter
from app.services.embedding import EmbeddingService
from app.core.config import settings
from app.utils.logger import logger
from app.utils.file_handler import generate_document_id

class RAGPipeline:
    """RAG Pipeline 관리 서비스 (문서 처리, 벡터 저장, 검색)"""

    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

        self.metadata_file = os.path.join(settings.VECTOR_DB_PATH, "metadata.json")
        self.documents_metadata = {}  # document_id -> metadata
    
    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"메타데이터 로드 실패: {e}")
        return {}
    
    def _save_metadata(self):
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.documents_metadata, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"메타데이터 저장 실패: {e}")


    def process_document(self, file_path: str, filename:str) -> str:
        """
        문서 전체 처리 파이프라인
        
        1. 파일 파싱
        2. 텍스트 청킹
        3. 임베딩 생성
        4. 벡터 DB 저장
        """
        try:
            text, page_numbers = self.loader.load_document(file_path)

            # 텍스트 청킹
            chunks = self.splitter.split_text(text, page_numbers)

            if not chunks:
                raise ValueError("청크 생성 실패: 생성된 청크가 없습니다")
            logger.info(f"청크 생성 완료: {len(chunks)} 청크")

            # 임베딩 생성
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.embed(chunk_texts)

            # 문서 ID 생성
            document_id = generate_document_id()
            file_size = os.path.getsize(file_path)
            upload_time = datetime.now().isoformat()

            self.documents_metadata[document_id] = {
                "document_id": document_id,
                "filename": filename,
                "file_size": file_size,
                "upload_time": upload_time,
                "total_chunks": len(chunks)
            }
            #메타데이터 저장
            self._save_metadata()

            # 벡터 DB 저장
            self.vector_store.add_documents(
                document_id = document_id, 
                embeddings = embeddings.tolist(), 
                chunks = chunks
                )
            logger.info(f"문서 처리 및 저장 완료:  {document_id}")
            return {
                "document_id": document_id,
                "filename": filename,
                "total_chunks": len(chunks),
                "file_size": file_size
            }


        except Exception as e:
            logger.error(f"문서 처리 오류: {str(e)}")
            raise

    def get_documents_metadata(self) -> List[Dict]:
        """저장된 문서 메타데이터 반환"""
        return list(self.documents_metadata.values())
    
    def get_document_metadata(self, document_id: str) -> Optional[Dict]:
        """특정 문서 메타데이터 반환"""
        return self.documents_metadata.get(document_id)