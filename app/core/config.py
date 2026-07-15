from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os


class Settings(BaseSettings):
    APP_NAME: str = "RAG Server"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    OPENAI_API_KEY: str = ""
    OPENAI_CHAT_MODEL: str = "gpt-3.5-turbo"

    # "openai" | "ollama" | "gemini" | "upstage" | "anthropic" 중 선택.
    # llm.py의 get_llm()이 이 값을 보고 적절한 클라이언트를 생성한다.
    LLM_PROVIDER: str = "upstage"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"

    UPSTAGE_API_KEY: str = ""
    UPSTAGE_CHAT_MODEL: str = "solar-pro"

    GEMINI_API_KEY: str = ""
    GEMINI_CHAT_MODEL: str = "gemini-2.0-flash"

    EMBEDDING_PROVIDER: str = "huggingface"  # "huggingface"(무료, 로컬) 또는 "openai"(유료, API)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    VECTOR_DB_PATH: str = "app/db/vector_db"

    UPLOAD_DIR: str = "data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 50

    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 128

    MAX_HISTORY_MESSAGES: int = 10

    TOP_K: int = 4
    SCORE_THRESHOLD: float = 0.3

    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"

    GRAPH_EXTRACTION_MODEL: str = ""
    GRAPH_COMMUNITY_DETECTION: bool = True
    GRAPH_TRAVERSAL_DEPTH: int = 2

    # "vector" | "graph" | "hybrid" 중 선택
    SEARCH_MODE: str = "hybrid"
    GRAPH_WEIGHT: float = 0.4
    VECTOR_WEIGHT: float = 0.6

    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_CHAT_MODEL: str = "claude-sonnet-4-20250514"

    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "rag-server"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_TRACING_ENABLED: bool = True

    LANGGRAPH_MAX_RETRIES: int = 3
    LANGGRAPH_ROUTING_ENABLED: bool = True

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()

# 서버 시작 시 필요한 디렉토리를 미리 생성. 없으면 첫 업로드 시 에러가 발생한다.
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)


def configure_langsmith():
    """LangSmith 트레이싱을 위한 환경변수를 설정한다."""
    # getattr: 구버전 .env에 해당 키가 없을 경우에도 안전하게 기본값을 반환한다.
    tracing_enabled = getattr(settings, "LANGSMITH_TRACING_ENABLED", False)
    api_key = getattr(settings, "LANGSMITH_API_KEY", "")
    project = getattr(settings, "LANGSMITH_PROJECT", "rag-server")
    endpoint = getattr(settings, "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    if tracing_enabled and api_key:
        # LangChain은 settings 객체가 아닌 os.environ을 직접 참조해 LangSmith를 활성화한다.
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


configure_langsmith()
