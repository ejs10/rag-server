from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    APP_NAME: str = "RAG Server"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # OpenAI 설정
    OPENAI_API_KEY: str = ""
    OPENAI_CHAT_MODEL: str = "gpt-3.5-turbo"
    
    # LLM 제공자 (openai, ollama, gemini)
    LLM_PROVIDER: str = "upstage" 
    
    # Ollama 설정 (LLM_PROVIDER=ollama일 때)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"

    # Upstage
    UPSTAGE_API_KEY: str = ""
    UPSTAGE_CHAT_MODEL: str = "solar-pro"

    # Gemini 설정
    GEMINI_API_KEY: str = ""
    GEMINI_CHAT_MODEL: str = "gemini-2.0-flash"

    # 임베딩 제공자 (openai, huggingface)
    EMBEDDING_PROVIDER: str = "huggingface"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # 벡터 DB 경로
    VECTOR_DB_PATH: str = "app/db/vector_db"
    
    # 파일 업로드 설정
    UPLOAD_DIR: str = "data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 50  # 최대 50MB
    
    # 청킹 설정
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 128
    
    # 대화 설정
    MAX_HISTORY_MESSAGES: int = 10
    
    # RAG 검색 설정
    TOP_K: int = 4
    SCORE_THRESHOLD: float = 0.3

    class Config:
        env_file = ".env"
        case_sensitive = True



settings = Settings()

Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
    