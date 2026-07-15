import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from app.models.schemas import (
    QueryRequest, QueryResponse, Source,
    LangGraphQueryRequest, LangGraphQueryResponse,
    CreateDatasetRequest, CreateDatasetResponse,
    RunEvalRequest, RunEvalResponse, FeedbackRequest,
)
from app.services.llm import LLMService
from app.services.conversation import conversation_manager
from app.services.rag_pipeline import shared_rag_pipeline
from app.core.config import settings
from app.utils.logger import logger

router = APIRouter(prefix="/chat", tags=["chat"])

# shared_rag_pipeline이 이미 로드한 임베딩 모델을 재사용한다.
# 별도로 EmbeddingService()를 생성하면 SentenceTransformer 모델이 중복 로드되어
# 메모리와 시작 시간을 낭비한다.
embedding_service = shared_rag_pipeline.embedding_service
llm_service = LLMService()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    기본 RAG 질문-답변 엔드포인트.
    질문 임베딩 → 유사 청크 검색 → LLM 답변 생성 → 히스토리 저장 순서로 처리한다.

    Parameters:
        request: 질문, 세션 ID, 문서 필터, top_k를 포함하는 요청 객체
    """
    try:
        question_embedding = embedding_service.embed([request.question])[0]

        similar_chunks = await run_in_threadpool(
            shared_rag_pipeline.vector_store.search,
            query_embedding=question_embedding.tolist() if hasattr(question_embedding, "tolist") else question_embedding,
            top_k=request.top_k,
            document_id=request.document_id
        )

        if not similar_chunks:
            # 빈 컨텍스트를 LLM에 보내면 hallucination이 유발될 수 있으므로 조기 반환한다.
            answer = "죄송합니다. 질문과 관련된 정보를 찾을 수 없습니다."
            sources = []
        else:
            context = "\n---\n".join([
                f"[출처: {r['document_id']}]\n{r['text']}"
                for r in similar_chunks
            ])

            chat_history = conversation_manager.get_conversation(request.session_id)

            answer = await run_in_threadpool(
                llm_service.generate_answer,
                question=request.question,
                context=context,
                chat_history=chat_history
            )

            sources = [
                Source(
                    document_id=r["document_id"],
                    chunk_index=r["chunk_index"],
                    score=r["score"],
                    text=r["text"][:200],
                    page=r.get("page"),
                    filename=shared_rag_pipeline.documents_metadata.get(r["document_id"], {}).get("filename"),
                )
                for r in similar_chunks
            ]

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
        # LLM API 속도 제한을 429로 변환해 프론트엔드가 재시도 안내를 보여줄 수 있게 한다.
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            logger.warning(f"LLM API rate limit exceeded: {e}")
            raise HTTPException(status_code=429, detail="LLM API의 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
        # LLM 제공자 서버의 일시적 과부하(get_llm의 자동 재시도로도 해결되지 않은 경우).
        if "503" in str(e) or "UNAVAILABLE" in str(e):
            logger.warning(f"LLM API temporarily unavailable: {e}")
            raise HTTPException(status_code=503, detail="LLM 서비스가 일시적으로 응답할 수 없습니다. 잠시 후 다시 시도해주세요.")
        logger.error(f"질문 처리 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="질문 처리 중 오류가 발생했습니다.")


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    SSE(Server-Sent Events) 방식으로 LLM 토큰을 실시간 스트리밍한다.
    단방향 서버→클라이언트 전송에는 WebSocket보다 SSE가 단순하고 적합하다.

    이벤트 타입:
        sources → 검색된 출처 목록 (스트리밍 시작 직전)
        chunk   → LLM 생성 텍스트 조각
        done    → 스트리밍 종료
        error   → 오류 발생
    """
    async def event_generator():
        try:
            question_embedding = await run_in_threadpool(
                embedding_service.embed_single, request.question
            )

            similar_chunks = await run_in_threadpool(
                shared_rag_pipeline.vector_store.search,
                query_embedding=question_embedding,
                top_k=request.top_k,
                document_id=request.document_id,
            )

            if not similar_chunks:
                yield f'data: {json.dumps({"type": "chunk", "text": "죄송합니다. 질문과 관련된 정보를 찾을 수 없습니다."}, ensure_ascii=False)}\n\n'
                yield f'data: {json.dumps({"type": "done"})}\n\n'
                return

            # 텍스트 스트리밍과 동시에 출처 패널을 렌더링할 수 있도록 출처를 먼저 전송한다.
            sources = [
                {
                    "document_id": r["document_id"],
                    "chunk_index": r["chunk_index"],
                    "score": r["score"],
                    "text": r["text"][:200],
                    "page": r.get("page"),
                    "filename": shared_rag_pipeline.documents_metadata.get(r["document_id"], {}).get("filename"),
                }
                for r in similar_chunks
            ]
            yield f'data: {json.dumps({"type": "sources", "data": sources}, ensure_ascii=False)}\n\n'

            context = "\n---\n".join([
                f"[출처: {r['document_id']}]\n{r['text']}" for r in similar_chunks
            ])
            chat_history = conversation_manager.get_conversation(request.session_id)

            from app.services.llm import build_rag_messages
            from app.services.rag_pipeline import langchain_llm

            # LLMService.generate_answer와 동일한 메시지 조립 로직을 공유한다.
            messages = build_rag_messages(request.question, context, chat_history)

            assembled_answer = ""
            async for chunk in langchain_llm.astream(messages):
                if chunk.text:
                    assembled_answer += chunk.text
                    yield f'data: {json.dumps({"type": "chunk", "text": chunk.text}, ensure_ascii=False)}\n\n'

            conversation_manager.add_message(request.session_id, "user", request.question)
            conversation_manager.add_message(request.session_id, "assistant", assembled_answer)

            yield f'data: {json.dumps({"type": "done"})}\n\n'

        except Exception as e:
            logger.error(f"스트리밍 오류: {str(e)}")
            yield f'data: {json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Nginx 리버스 프록시 버퍼링 비활성화
        },
    )


@router.post("/query/langgraph", response_model=LangGraphQueryResponse)
async def query_langgraph(request: LangGraphQueryRequest):
    """
    LangGraph 워크플로우 기반 질문-답변 엔드포인트.
    쿼리 재작성 → 라우팅 → 검색 → 관련성 필터 → 답변 생성 → 품질 평가 단계를 거친다.

    Parameters:
        request: LangGraphQueryRequest 형식의 요청 객체
    """
    try:
        # 지연 임포트: 최상단 임포트 시 서버 시작 시간이 길어질 수 있어 호출 시점에 로드한다.
        from app.services.rag_pipeline import run_rag_workflow

        result = await run_in_threadpool(
            run_rag_workflow,
            question=request.question,
            session_id=request.session_id,
            document_id=request.document_id,
            top_k=request.top_k,
            use_rerank=request.use_rerank,
            use_query_rewrite=request.use_query_rewriting,
        )

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
        if "503" in str(e) or "UNAVAILABLE" in str(e):
            raise HTTPException(
                status_code=503,
                detail="LLM 서비스가 일시적으로 응답할 수 없습니다. 잠시 후 다시 시도해주세요."
            )
        raise HTTPException(
            status_code=500,
            detail=f"LangGraph 질문 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """
    특정 세션의 대화 기록을 초기화한다.
    프론트엔드의 "대화 초기화" 버튼에서 호출한다.

    Parameters:
        session_id: 초기화할 세션 ID
    """
    conversation_manager.clear_conversation(session_id)
    return {"status": "success", "session_id": session_id}


eval_router = APIRouter(prefix="/eval", tags=["evaluation"])


@eval_router.post("/datasets", response_model=CreateDatasetResponse)
async def create_dataset(request: CreateDatasetRequest):
    """평가 데이터셋 생성 (POST /eval/datasets)"""
    try:
        from app.services.evaluation import create_eval_dataset
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
    """평가 데이터셋 목록 조회 (GET /eval/datasets)"""
    try:
        from app.services.evaluation import list_eval_datasets
        datasets = await run_in_threadpool(list_eval_datasets)
        return {"datasets": datasets, "count": len(datasets)}
    except Exception as e:
        logger.error(f"데이터셋 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@eval_router.post("/run", response_model=RunEvalResponse)
async def execute_evaluation(request: RunEvalRequest):
    """평가 실행 (POST /eval/run)"""
    try:
        from app.services.evaluation import run_evaluation
        result = await run_in_threadpool(
            run_evaluation,
            dataset_name=request.dataset_name,
            experiment_prefix=request.experiment_prefix,
        )
        if result is None:
            # run_evaluation이 None을 반환하면 LangSmith 클라이언트를 초기화할 수 없는 상태다.
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
    """사용자 피드백 제출 (POST /eval/feedback)"""
    try:
        from app.services.evaluation import log_feedback
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
