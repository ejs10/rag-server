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

    # Neo4j 설정
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"

    # GraphRAG 설정
    GRAPH_EXTRACTION_MODEL: str = ""  # 엔티티 추출에 사용할 LLM (비워두면 기본 LLM 사용)
    GRAPH_COMMUNITY_DETECTION: bool = True  # 커뮤니티 탐지 활성화
    GRAPH_TRAVERSAL_DEPTH: int = 2  # 그래프 순회 깊이

    # 검색 모드
    SEARCH_MODE: str = "hybrid"  # "vector", "graph", "hybrid"
    GRAPH_WEIGHT: float = 0.4  # 하이브리드 검색 시 그래프 결과 가중치
    VECTOR_WEIGHT: float = 0.6  # 하이브리드 검색 시 벡터 결과 가중치

    # Anthropic (Claude) 추가
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_CHAT_MODEL: str = "claude-sonnet-4-20250514"

    # [추가] LangSmith 설정
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "rag-server"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_TRACING_ENABLED: bool = True

    #LangChain 설정
    LANGCHAIN_TRACING_V2: bool = True  # LangSmith 트레이싱 활성화
    LANGCHAIN_PROJECT: str = "rag-server"

    #LangGraph 설정
    LANGGRAPH_MAX_RETRIES: int = 3  # 최대 재시도 횟수
    LANGGRAPH_ROUTING_ENABLED: bool = True  # 라우팅 활성화

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)

# [추가] LangSmith 환경변수 자동 설정

def configure_langsmith():
    """LangSmith 트레이싱을 위한 환경변수 설정"""
    # getattr 사용 → 필드가 누락되어도 에러 없이 기본값 반환
    tracing_enabled = getattr(settings, "LANGSMITH_TRACING_ENABLED", False)
    api_key = getattr(settings, "LANGSMITH_API_KEY", "")
    project = getattr(settings, "LANGSMITH_PROJECT", "rag-server")
    endpoint = getattr(settings, "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
 
    if tracing_enabled and api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
 
configure_langsmith()
    