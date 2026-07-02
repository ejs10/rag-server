from typing import List, Dict, Optional
import chromadb
from app.core.config import settings as app_settings
from app.utils.logger import logger


class VectorStore:
    """ChromaDB를 이용한 벡터 임베딩 저장 및 유사도 검색 서비스."""

    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=app_settings.VECTOR_DB_PATH)
            self.collection_name = "documents"
            # hnsw:space=cosine: 벡터 방향 기반 유사도. 크기에 무관해 텍스트 임베딩에 적합하다.
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
        문서 청크와 임베딩을 ChromaDB에 저장.

        Parameters:
            document_id: 문서 고유 ID
            embeddings : 각 청크의 벡터 (청크 수 × 임베딩 차원)
            chunks     : 청크 딕셔너리 목록 [{"text": "...", "page": 3}, ...]
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
        쿼리 벡터와 가장 유사한 청크를 검색.

        Parameters:
            query_embedding: 질문을 변환한 벡터
            top_k          : 반환할 최대 결과 수
            document_id    : 지정 시 해당 문서 내에서만 검색

        Returns:
            유사도 순으로 정렬된 청크 딕셔너리 목록
        """
        try:
            where_filter = None
            if document_id:
                where_filter = {"document_id": {"$eq": document_id}}

            results = self.collection.query(
                query_embeddings=[query_embedding],  # 배치 쿼리 형식을 위해 리스트로 감싼다.
                n_results=top_k,
                where=where_filter,
                include=["embeddings", "documents", "metadatas", "distances"]
            )

            search_results = []
            if results["documents"] and len(results["documents"][0]) > 0:
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i]
                    # ChromaDB 코사인 거리(0~2)를 유사도(0~1)로 변환한다.
                    score = max(0, 1 - distance)
                    if score < app_settings.SCORE_THRESHOLD:
                        continue
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

    def delete_document(self, document_id: str) -> None:
        """
        특정 문서의 모든 청크를 ChromaDB에서 삭제.

        Parameters:
            document_id: 삭제할 문서의 고유 ID
        """
        try:
            self.collection.delete(
                where={"document_id": {"$eq": document_id}}
            )
            logger.info(f"벡터 DB 문서 삭제 완료: {document_id}")
        except Exception as e:
            logger.error(f"벡터 DB 문서 삭제 오류: {str(e)}")
            raise

    def get_documents_list(self) -> List[Dict]:
        """
        저장된 문서 목록과 청크 수를 반환.
        문서 메타데이터 API는 RAGPipeline.metadata를 사용하므로 이 메서드는 ChromaDB 직접 확인.
        """
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
