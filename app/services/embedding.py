from typing import List
import numpy as np
from app.core.config import settings
from app.utils.logger import logger

class EmbeddingService:
    """임베딩 생성 서비스"""

    def __init__(self):
        self.provider = settings.EMBEDDING_PROVIDER
        self.model = None  # 모델은 필요에 따라 설정
        self._initialize_model()

    def _initialize_model(self):
        """모델 초기화 (예: Hugging Face 모델 로드)"""
        try:
            if self.provider == "huggingface":
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
                logger.info(f"HuggingFace 모델 로드: {settings.EMBEDDING_MODEL}")
            
            elif self.provider == "openai":
                import openai
                openai.api_key = settings.OPENAI_API_KEY
                self.model = "openai"
                logger.info("OpenAI 임베딩 모델 준비")

            else:
                raise ValueError(f"지원되지 않는 임베딩 제공자: {self.provider}")
            
        except Exception as e:
            logger.error(f"임베딩 모델 초기화 오류: {str(e)}")
            raise

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        텍스트 리스트를 임베딩으로 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
        
        Returns:
            임베딩 배열 (n_texts, embedding_dim)
        """

        try:
            if self.provider == "huggingface":
                embeddings = self.model.encode(texts)
                return np.array(embeddings)
            
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
        """단일 텍스트 임베딩"""
        embeddings = self.embed([text])
        return embeddings[0].tolist()