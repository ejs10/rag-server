from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.models.schemas import (
    QueryRequest, QueryResponse, Source,
    
    LangGraphQueryRequest, LangGraphQueryResponse,
    CreateDatasetRequest, CreateDatasetResponse,
    RunEvalRequest, RunEvalResponse, FeedbackRequest,
)
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.conversation import conversation_manager
from app.services.rag_pipeline import shared_rag_pipeline
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


# [추가] LangGraph 워크플로우 기반 질의 엔드포인트
@router.post("/query/langgraph", response_model=LangGraphQueryResponse)
async def query_langgraph(request: LangGraphQueryResponse):
    """
    LangGraph 워크플로우 기반 질문 답변 생성
 
    - 쿼리 분석 → 쿼리 재작성 → 검색 → (선택) 리랭킹 → 답변 생성
    - LangSmith 트레이싱 자동 연동
    """
    try:
        from app.services.rag_pipeline import run_rag_workflow
 
        result = await run_in_threadpool(
            run_rag_workflow,
            question=request.question,
            session_id=request.session_id,
            document_id=request.document_id,
            top_k=request.top_k,
            use_rerank=request.use_rerank,
            use_query_rewrite=request.use_query_rewrite,
        )
 
        # 대화 히스토리 저장
        conversation_manager.add_message(
            session_id=request.session_id,
            role="user",
            content=request.question
        )
        conversation_manager.add_message(
            session_id=request.session_id,
            role="assistant",
            content=result["answer"]
        )
 
        sources = [
            Source(
                document_id=s["document_id"],
                chunk_index=s["chunk_index"],
                score=s["score"],
                text=s["text"][:200],
                page=s.get("page"),
            )
            for s in result.get("sources", [])
        ]
 
        return LangGraphQueryResponse(
            answer=result["answer"],
            sources=sources,
            session_id=request.session_id,
            rewritten_query=result.get("rewritten_query"),
            route=result.get("route"),
            node_trace=result.get("node_trace"),
        )
 
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LangGraph 질문 처리 오류: {str(e)}")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            raise HTTPException(
                status_code=429,
                detail="LLM API의 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
            )
        raise HTTPException(
            status_code=500,
            detail=f"LangGraph 질문 처리 중 오류가 발생했습니다: {str(e)}"
        )

# [추가] LangSmith 평가 엔드포인트 (기존 eval.py 통합)
eval_router = APIRouter(prefix="/eval", tags=["evaluation"])
@eval_router.post("/datasets", response_model=CreateDatasetResponse)
async def create_dataset(request: CreateDatasetRequest):
    """평가 데이터셋 생성"""
    try:
        from app.services.rag_pipeline import create_eval_dataset
        examples = [ex.model_dump() for ex in request.examples]
        dataset_id = await run_in_threadpool(
            create_eval_dataset,
            dataset_name=request.dataset_name,
            examples=examples,
            description=request.description,
        )
        return CreateDatasetResponse(
            dataset_id=dataset_id,
            dataset_name=request.dataset_name,
            example_count=len(request.examples),
            status="created" if dataset_id else "failed",
        )
    except Exception as e:
        logger.error(f"데이터셋 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
@eval_router.get("/datasets")
async def get_datasets():
    """평가 데이터셋 목록 조회"""
    try:
        from app.services.rag_pipeline import list_eval_datasets
        datasets = await run_in_threadpool(list_eval_datasets)
        return {"datasets": datasets, "count": len(datasets)}
    except Exception as e:
        logger.error(f"데이터셋 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
@eval_router.post("/run", response_model=RunEvalResponse)
async def execute_evaluation(request: RunEvalRequest):
    """평가 실행"""
    try:
        from app.services.rag_pipeline import run_evaluation
        result = await run_in_threadpool(
            run_evaluation,
            dataset_name=request.dataset_name,
            experiment_prefix=request.experiment_prefix,
        )
        if result is None:
            return RunEvalResponse(
                status="skipped",
                experiment_prefix=request.experiment_prefix,
                dataset_name=request.dataset_name,
                error="LangSmith 클라이언트를 사용할 수 없습니다",
            )
        return RunEvalResponse(
            status=result.get("status", "unknown"),
            experiment_prefix=request.experiment_prefix,
            dataset_name=request.dataset_name,
            error=result.get("error"),
        )
    except Exception as e:
        logger.error(f"평가 실행 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
@eval_router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """LangSmith에 피드백 제출"""
    try:
        from app.services.rag_pipeline import log_feedback
        success = await run_in_threadpool(
            log_feedback,
            run_id=request.run_id,
            key=request.key,
            score=request.score,
            comment=request.comment,
        )
        return {"success": success}
    except Exception as e:
        logger.error(f"피드백 제출 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))