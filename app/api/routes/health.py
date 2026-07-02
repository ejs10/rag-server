from fastapi import APIRouter
from app.core.config import settings
from app.models.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    서버 상태 및 현재 설정을 반환한다. 모니터링 시스템의 헬스체크 엔드포인트로 사용된다.
    """
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment={
            "llm_provider": settings.LLM_PROVIDER,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "debug": settings.DEBUG,
            # getattr: 구버전 .env에 해당 키가 없을 경우를 대비한 안전한 접근
            "langsmith_tracing": getattr(settings, "LANGSMITH_TRACING_ENABLED", False)
                                and bool(getattr(settings, "LANGSMITH_API_KEY", "")),
            "langsmith_project": getattr(settings, "LANGSMITH_PROJECT", ""),
            "langgraph_routing": getattr(settings, "LANGGRAPH_ROUTING_ENABLED", False),
        }
    )
