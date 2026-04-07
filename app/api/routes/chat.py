from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.models.schemas import QueryRequest, QueryResponse, Source
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.conversation import conversation_manager
from app.services.rag_pipeline import shared_rag_pipeline  # 공통 파이프라인 가져오기
from app.core.config import settings
from app.utils.logger import logger

router = APIRouter(prefix="/chat", tags=["chat"])
embedding_service = EmbeddingService()
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
        similar_chunks = await run_in_threadpool(
            shared_rag_pipeline.vector_store.search,
            query_embedding=question_embedding.tolist() if hasattr(question_embedding, "tolist") else question_embedding,
            top_k=request.top_k,
            document_id=request.document_id  # 문서 ID 기반 검색 추가
        )

        if not similar_chunks:
            answer = "죄송합니다. 질문과 관련된 정보를 찾을 수 없습니다."
            sources = []
        else:
            # 3. 검색 결과로 컨텍스트 구성
            context = "\n---\n".join([
                f"[출처: {r['document_id']}]\n{r['text']}"
                for r in similar_chunks
            ])
            #대화 히스토리 가져오기
            chat_history = conversation_manager.get_conversation(request.session_id)

            # LLM 답변 생성
            answer = await run_in_threadpool(
                llm_service.generate_answer,
                question=request.question,
                context=context,
                chat_history=chat_history
            )
            
            #출처 정보 구성
            sources = [
                Source(
                    document_id=r["document_id"],
                    chunk_index=r["chunk_index"],
                    score=r["score"],
                    text=r["text"][:200],  # 텍스트 일부만 반환,
                    page=r.get("page"),
                )
                for r in similar_chunks
            ]

        # 대화 히스토리 저장
        conversation_manager.add_message(
            session_id=request.session_id,
            role="user",
            content=request.question
        )
        conversation_manager.add_message(
            session_id=request.session_id,
            role="assistant",
            content=answer
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=request.session_id
        )


    except HTTPException:
        raise
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            logger.warning(f"LLM API rate limit exceeded: {e}")
            raise HTTPException(status_code=429, detail="LLM API의 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
        raise HTTPException(status_code=500, detail="질문 처리 중 오류가 발생했습니다.")
    