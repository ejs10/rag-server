from fastapi import APIRouter
from app.core.config import settings
from app.models.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/", response_model=HealthResponse)
async def health_check():
    """헬스 체크 엔드포인트"""
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment={
            "llm_provider": settings.LLM_PROVIDER,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "debug": settings.DEBUG,
            # [추가] LangChain/LangSmith/LangGraph 상태
            "langsmith_tracing": getattr(settings, "LANGSMITH_TRACING_ENABLED", False)
                                and bool(getattr(settings, "LANGSMITH_API_KEY", "")),
            "langsmith_project": getattr(settings, "LANGSMITH_PROJECT", ""),
            "langgraph_routing": getattr(settings, "LANGGRAPH_ROUTING_ENABLED", False),
        }
    )