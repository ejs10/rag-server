from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import health, documents, upload, chat
from app.utils.logger import logger

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    description="LLM 기반 문서 Q&A 서버 (RAG)"
)

# allow_origins=["*"]: 개발 편의용 전체 허용. 운영 환경에서는 특정 도메인으로 제한 권장.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.on_event("startup")
async def startup_event():
    """
    서버 시작 시 실행되는 초기화 핸들러.
    """
    logger.info(f"=== {settings.APP_NAME} 시작 ===")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
    logger.info(f"Embedding Provider: {settings.EMBEDDING_PROVIDER}")
    logger.info(f"Vector DB Path: {settings.VECTOR_DB_PATH}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"=== {settings.APP_NAME} 종료 ===")


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
