from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, Source
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm import LLMService
from app.services.conversation import conversation_manager
from app.core.config import settings
from app.utils.logger import logger

router = APIRouter(prefix="/chat", tags=["chat"])
embedding_service = EmbeddingService()
vector_store = VectorStore()
llm_service = LLMService()

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    질문에 답변 생성
    
    1. 질문 임베딩
    2. 유사한 문서 청크 검색
    3. 검색 결과 + 질문으로 LLM 답변 생성
    4. 대화 히스토리 저장
    """
    try:
        # 질문 임베딩
        question_embedding = embedding_service.embed([request.question])[0]

        # 유사한 문서 청크 검색
        similar_chunks = vector_store.similarity_search(
            query_embedding=question_embedding,
            top_k=settings.TOP_K,
            document_id=request.document_id
        )

        if not similar_chunks:
            raise HTTPException(status_code=404, detail="유사한 문서 청크를 찾을 수 없습니다.")

        # LLM 답변 생성
        context = "\n\n".join([f"Page {chunk['page']}:\n{chunk['text']}" for chunk in similar_chunks])
        llm_response = llm_service.generate_answer(
            question=request.query,
            context=context
        )

        # 대화 히스토리 저장
        conversation_manager.save_conversation(
            question=request.query,
            answer=llm_response,
            sources=[Source(page=chunk["page"], text=chunk["text"]) for chunk in similar_chunks]
        )

        return QueryResponse(
            answer=llm_response,
            sources=[Source(page=chunk["page"], text=chunk["text"]) for chunk in similar_chunks]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"질문 처리 실패: {e}")
        raise HTTPException(status_code=500, detail="질문 처리 중 오류가 발생했습니다.")
    
    