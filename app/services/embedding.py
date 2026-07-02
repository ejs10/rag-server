from typing import List
import numpy as np
from app.core.config import settings
from app.utils.logger import logger

class EmbeddingService:
    """임베딩 생성 서비스 - 텍스트를 벡터로 변환"""

    def __init__(self):
        self.provider = settings.EMBEDDING_PROVIDER
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            if self.provider == "huggingface":
                from sentence_transformers import SentenceTransformer
                # 요청마다 로드하면 매번 수 초가 걸리므로 서버 시작 시 한 번만 로드한다.
                self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
                logger.info(f"HuggingFace 모델 로드: {settings.EMBEDDING_MODEL}")

            elif self.provider == "openai":
                import openai
                openai.api_key = settings.OPENAI_API_KEY
                # API 호출 방식이라 모델 객체가 없다. provider 확인용 플래그로만 사용.
                self.model = "openai"
                logger.info("OpenAI 임베딩 모델 준비")

            else:
                raise ValueError(f"지원되지 않는 임베딩 제공자: {self.provider}")

        except Exception as e:
            logger.error(f"임베딩 모델 초기화 오류: {str(e)}")
            raise

    def embed(self, texts: List[str]) -> np.ndarray:
        """텍스트 리스트를 임베딩 배열로 변환. shape: (텍스트 수, 임베딩 차원)"""
        try:
            if self.provider == "huggingface":
                embeddings = self.model.encode(texts)
                return np.array(embeddings)  # dtype 일관성 보장

            elif self.provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=settings.OPENAI_API_KEY)
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts
                )
                embeddings = [item.embedding for item in response.data]
                return np.array(embeddings)

        except Exception as e:
            logger.error(f"임베딩 생성 오류: {str(e)}")
            raise

    def embed_single(self, text: str) -> List[float]:
        """단일 텍스트 임베딩 - 주로 질문 벡터화에 사용"""
        embeddings = self.embed([text])
        return embeddings[0].tolist()


from langchain_core.embeddings import Embeddings as LangChainEmbeddingsBase

class LangChainEmbeddingsWrapper(LangChainEmbeddingsBase):
    """
    EmbeddingService를 LangChain Embeddings 인터페이스로 감싸는 어댑터.
    LangGraph/LangChain VectorStore는 embed_documents, embed_query 시그니처를 요구한다.
    """

    def __init__(self, embedding_service: EmbeddingService):
        self._service = embedding_service

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self._service.embed(texts)
        return result.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._service.embed_single(text)
