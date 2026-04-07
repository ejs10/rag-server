from typing import List, Dict, Optional
import chromadb
from app.core.config import settings as app_settings
from app.utils.logger import logger

class VectorStore:
    """벡터 DB (Chroma) 관리 서비스"""
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=app_settings.VECTOR_DB_PATH)
            self.collection_name = "documents"
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("벡터 DB 초기화 완료")
        except Exception as e:
            logger.error(f"벡터 DB 초기화 오류: {str(e)}")
            raise

    def add_documents(self, document_id: str, embeddings: List[List[float]], 
                      chunks: List[Dict]) -> None:
        """
        문서 청크와 임베딩 저장
        
        Args:
            document_id: 문서 ID
            embeddings: 임베딩 리스트
            chunks: 청크 리스트 (text, page 포함)
        """
        try:
            ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            documents = [chunk["text"] for chunk in chunks]
            metadatas = [
                {
                    "document_id": document_id,
                    "chunk_index": i,
                    "page": chunk.get("page", 1)
                } for i, chunk in enumerate(chunks)
            ]
            self.collection.add(
                ids=ids, 
                embeddings=embeddings, 
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"문서 저장 완료: {document_id}, 청크 수: {len(chunks)}")
        except Exception as e:
            logger.error(f"문서 저장 오류: {str(e)}")
            raise
    def search(self, query_embedding: List[float], top_k: int = 4,
               document_id: Optional[str] = None) -> List[Dict]:
        """
        쿼리 임베딩과 유사한 문서 청크 검색
        
        Args:
            query_embedding: 검색할 쿼리 임베딩
            top_k: 반환할 유사한 청크 수
        
        Returns:
            유사한 청크 리스트 (text, page 포함)
        """
        try:

            where_filter = None
            if document_id:
                where_filter = {"document_id": {"$eq": document_id}}
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["embeddings", "documents", "metadatas", "distances"]
            )
            search_results = []
            if results["documents"] and len(results["documents"][0]) > 0:
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i]
                    score = max(0, 1 - distance)  # 유사도를 거리에서 변환 (0~1)
                    
                    metadata = results["metadatas"][0][i]
                    search_results.append({
                        "text": doc,
                        "document_id": metadata.get("document_id"),
                        "chunk_index": metadata.get("chunk_index"),
                        "page": metadata.get("page", 1),
                        "score": score
                    })
            return search_results
        except Exception as e:
            logger.error(f"검색 오류: {str(e)}")
            raise

    def get_documents_list(self) -> List[Dict]:
        try:
            results = self.collection.get(include=["metadatas"])

            doc_dict = {}
            for metadata in results["metadatas"]:
                doc_id = metadata["document_id"]
                if doc_id not in doc_dict:
                    doc_dict[doc_id] = {"chunk_count": 0}
                doc_dict[doc_id]["chunk_count"] += 1
            
            return [
                {"document_id": doc_id, "chunk_count": info["chunk_count"]}
                for doc_id, info in doc_dict.items()
            ]
        except Exception as e:
            logger.error(f"문서 목록 조회 오류: {str(e)}")
            raise