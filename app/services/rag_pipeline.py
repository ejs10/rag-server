from typing import List, Dict, Optional, Any, Callable, TypedDict  # [수정] TypedDict 등 추가
import os
from datetime import datetime
import json
from app.services.document_loader import DocumentLoader
from app.services.vector_store import VectorStore
from app.services.text_splitter import TextSplitter
from app.services.embedding import EmbeddingService, LangChainEmbeddingsWrapper  # [수정] 래퍼 import 추가
from app.core.config import settings
from app.utils.logger import logger
from app.utils.file_handler import generate_document_id

class RAGPipeline:
    """RAG Pipeline 관리 서비스 (문서 처리, 벡터 저장, 검색)"""

    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

        self.metadata_file = os.path.join(settings.VECTOR_DB_PATH, "metadata.json")
        self.documents_metadata = self._load_metadata()  # document_id -> metadata
    
    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"메타데이터 로드 실패: {e}")
        return {}
    
    def _save_metadata(self):
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.documents_metadata, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"메타데이터 저장 실패: {e}")


    def process_document(self, file_path: str, filename:str) -> str:
        """
        문서 전체 처리 파이프라인
        
        1. 파일 파싱
        2. 텍스트 청킹
        3. 임베딩 생성
        4. 벡터 DB 저장
        """
        try:
            text, page_numbers = self.loader.load_document(file_path)

            # 텍스트 청킹
            chunks = self.splitter.split_text(text, page_numbers)

            if not chunks:
                raise ValueError("청크 생성 실패: 생성된 청크가 없습니다")
            logger.info(f"청크 생성 완료: {len(chunks)} 청크")

            # 임베딩 생성
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.embed(chunk_texts)

            # 문서 ID 생성
            document_id = generate_document_id()
            file_size = os.path.getsize(file_path)
            upload_time = datetime.now().isoformat()

            self.documents_metadata[document_id] = {
                "document_id": document_id,
                "filename": filename,
                "file_size": file_size,
                "upload_time": upload_time,
                "total_chunks": len(chunks)
            }
            #메타데이터 저장
            self._save_metadata()

            # 벡터 DB 저장
            self.vector_store.add_documents(
                document_id = document_id, 
                embeddings = embeddings.tolist(), 
                chunks = chunks
                )
            logger.info(f"문서 처리 및 저장 완료:  {document_id}")
            return {
                "document_id": document_id,
                "filename": filename,
                "total_chunks": len(chunks),
                "file_size": file_size
            }


        except Exception as e:
            logger.error(f"문서 처리 오류: {str(e)}")
            raise

    def get_documents_metadata(self) -> List[Dict]:
        """저장된 문서 메타데이터 반환"""
        return list(self.documents_metadata.values())
    
    def get_document_metadata(self, document_id: str) -> Optional[Dict]:
        """특정 문서 메타데이터 반환"""
        return self.documents_metadata.get(document_id)
    
shared_rag_pipeline = RAGPipeline()

#  [추가] LangChain 체인 (기존 langchain_chains.py 통합)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable
 
from app.services.llm import (
    LLMService, LangChainLLMWrapper,
    RAG_QA_PROMPT, QUERY_REWRITE_PROMPT, QUERY_ROUTING_PROMPT,
    ANSWER_GRADING_PROMPT, DOCUMENT_RELEVANCE_PROMPT,
)
from app.services.conversation import conversation_manager
 
# LangChain 래퍼 싱글톤
_lc_embedding_service = shared_rag_pipeline.embedding_service
_lc_llm_service = LLMService()
langchain_llm = LangChainLLMWrapper(llm_service=_lc_llm_service)
langchain_embeddings = LangChainEmbeddingsWrapper(embedding_service=_lc_embedding_service)
 
 
@traceable(name="query_rewrite")
def rewrite_query(question: str, chat_history: List[Dict] = None) -> str:
    """사용자 질문을 검색에 최적화된 형태로 재작성"""
    try:
        messages = []
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        chain = QUERY_REWRITE_PROMPT | langchain_llm | StrOutputParser()
        rewritten = chain.invoke({
            "question": question,
            "chat_history": messages if messages else [],
        })
        logger.info(f"쿼리 재작성: '{question}' → '{rewritten.strip()}'")
        return rewritten.strip()
    except Exception as e:
        logger.warning(f"쿼리 재작성 실패, 원본 사용: {e}")
        return question
 
 
@traceable(name="query_routing")
def route_query(question: str) -> str:
    """질문을 분석하여 검색 방식을 결정 (vector / graph / hybrid)"""
    if not settings.LANGGRAPH_ROUTING_ENABLED:
        return settings.SEARCH_MODE
    try:
        chain = QUERY_ROUTING_PROMPT | langchain_llm | StrOutputParser()
        route = chain.invoke({"question": question}).strip().lower()
        valid_routes = {"vector", "graph", "hybrid"}
        if route not in valid_routes:
            route = settings.SEARCH_MODE
        logger.info(f"쿼리 라우팅 결과: {route}")
        return route
    except Exception as e:
        logger.warning(f"쿼리 라우팅 실패, 기본값 사용: {e}")
        return settings.SEARCH_MODE
 
 
@traceable(name="vector_search")
def search_documents(
    query: str,
    top_k: int = 4,
    document_id: Optional[str] = None,
) -> List[Dict]:
    """기존 VectorStore를 활용한 벡터 검색"""
    try:
        query_embedding = _lc_embedding_service.embed_single(query)
        results = shared_rag_pipeline.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
        )
        logger.info(f"검색 완료: {len(results)}개 청크 반환")
        return results
    except Exception as e:
        logger.error(f"검색 오류: {e}")
        return []
 
 
@traceable(name="document_relevance_filter")
def filter_relevant_documents(
    question: str,
    documents: List[Dict],
) -> List[Dict]:
    """검색된 문서 중 질문과 관련있는 문서만 필터링"""
    if not documents:
        return []
    try:
        chain = DOCUMENT_RELEVANCE_PROMPT | langchain_llm | StrOutputParser()
        filtered = []
        for doc in documents:
            result = chain.invoke({
                "question": question,
                "document": doc["text"][:500],
            })
            if "yes" in result.strip().lower():
                filtered.append(doc)
        logger.info(f"관련성 필터링: {len(documents)}개 → {len(filtered)}개")
        return filtered if filtered else documents
    except Exception as e:
        logger.warning(f"관련성 필터링 실패, 원본 사용: {e}")
        return documents
 
 
@traceable(name="generate_answer")
def generate_rag_answer(
    question: str,
    context: str,
    chat_history: List[Dict] = None,
) -> str:
    """LangChain 체인을 통한 RAG 답변 생성"""
    try:
        messages = []
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        chain = RAG_QA_PROMPT | langchain_llm | StrOutputParser()
        answer = chain.invoke({
            "question": question,
            "context": context,
            "chat_history": messages if messages else [],
        })
        logger.info("LangChain RAG 답변 생성 완료")
        return answer
    except Exception as e:
        logger.error(f"LangChain RAG 답변 생성 오류: {e}")
        raise
 
 
@traceable(name="answer_grading")
def grade_answer(
    question: str,
    context: str,
    answer: str,
) -> Dict[str, Any]:
    """생성된 답변의 품질을 평가 (Self-RAG)"""
    try:
        chain = ANSWER_GRADING_PROMPT | langchain_llm | StrOutputParser()
        raw = chain.invoke({
            "question": question,
            "context": context,
            "answer": answer,
        })
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            scores = json.loads(cleaned)
        except json.JSONDecodeError:
            scores = {
                "faithfulness": 0.0, "relevance": 0.0,
                "hallucination": 0.0, "explanation": raw,
            }
        logger.info(f"답변 품질 평가: {scores}")
        return scores
    except Exception as e:
        logger.warning(f"답변 품질 평가 실패: {e}")
        return {
            "faithfulness": 0.0, "relevance": 0.0,
            "hallucination": 0.0, "explanation": str(e),
        }
 

#  [추가] LangGraph RAG 워크플로우 (기존 langgraph_workflow.py 통합)
from langgraph.graph import StateGraph, END
 
 
class RAGState(TypedDict):
    """LangGraph 워크플로우 상태"""
    question: str
    session_id: str
    document_id: Optional[str]
    top_k: int
    use_rerank: bool
    use_query_rewrite: bool
    rewritten_query: Optional[str]
    route: Optional[str]
    search_results: List[Dict]
    filtered_results: List[Dict]
    context: str
    chat_history: List[Dict]
    answer: str
    sources: List[Dict]
    grade_scores: Dict
    node_trace: List[str]
    retry_count: int
 
 
def initialize_node(state: RAGState) -> dict:
    logger.info("[LangGraph] 초기화 노드 실행")
    chat_history = conversation_manager.get_conversation(state["session_id"])
    return {
        "chat_history": chat_history, "node_trace": ["initialize"],
        "retry_count": 0, "rewritten_query": None, "route": None,
        "search_results": [], "filtered_results": [], "context": "",
        "answer": "", "sources": [], "grade_scores": {},
    }
 
 
def query_rewrite_node(state: RAGState) -> dict:
    logger.info("[LangGraph] 쿼리 재작성 노드 실행")
    rewritten = rewrite_query(
        question=state["question"],
        chat_history=state.get("chat_history", []),
    )
    trace = state.get("node_trace", []) + ["query_rewrite"]
    return {"rewritten_query": rewritten, "node_trace": trace}
 
 
def query_routing_node(state: RAGState) -> dict:
    logger.info("[LangGraph] 쿼리 라우팅 노드 실행")
    effective_query = state.get("rewritten_query") or state["question"]
    route = route_query(effective_query)
    trace = state.get("node_trace", []) + ["query_routing"]
    return {"route": route, "node_trace": trace}
 
 
def vector_search_node(state: RAGState) -> dict:
    logger.info("[LangGraph] 벡터 검색 노드 실행")
    effective_query = state.get("rewritten_query") or state["question"]
    results = search_documents(
        query=effective_query,
        top_k=state.get("top_k", 4),
        document_id=state.get("document_id"),
    )
    trace = state.get("node_trace", []) + ["vector_search"]
    return {"search_results": results, "node_trace": trace}
 
 
def relevance_filter_node(state: RAGState) -> dict:
    logger.info("[LangGraph] 관련성 필터링 노드 실행")
    effective_query = state.get("rewritten_query") or state["question"]
    filtered = filter_relevant_documents(
        question=effective_query,
        documents=state.get("search_results", []),
    )
    trace = state.get("node_trace", []) + ["relevance_filter"]
    return {"filtered_results": filtered, "node_trace": trace}
 
 
def build_context_node(state: RAGState) -> dict:
    logger.info("[LangGraph] 컨텍스트 구성 노드 실행")
    results = state.get("filtered_results") or state.get("search_results", [])
    if results:
        context = "\n---\n".join([
            f"[출처: {r['document_id']}, 페이지: {r.get('page', '?')}]\n{r['text']}"
            for r in results
        ])
        sources = results
    else:
        context = ""
        sources = []
    trace = state.get("node_trace", []) + ["build_context"]
    return {"context": context, "sources": sources, "node_trace": trace}
 
 
def generate_answer_node(state: RAGState) -> dict:
    logger.info("[LangGraph] 답변 생성 노드 실행")
    context = state.get("context", "")
    if not context:
        answer = "죄송합니다. 질문과 관련된 정보를 찾을 수 없습니다."
    else:
        answer = generate_rag_answer(
            question=state["question"],
            context=context,
            chat_history=state.get("chat_history", []),
        )
    trace = state.get("node_trace", []) + ["generate_answer"]
    return {"answer": answer, "node_trace": trace}
 
 
def grade_answer_node(state: RAGState) -> dict:
    logger.info("[LangGraph] 답변 품질 평가 노드 실행")
    context = state.get("context", "")
    answer = state.get("answer", "")
    if not context or not answer:
        scores = {"faithfulness": 0.0, "relevance": 0.0, "hallucination": 1.0}
    else:
        scores = grade_answer(
            question=state["question"], context=context, answer=answer,
        )
    trace = state.get("node_trace", []) + ["grade_answer"]
    return {"grade_scores": scores, "node_trace": trace}
 
 
def regenerate_node(state: RAGState) -> dict:
    logger.info("[LangGraph] 답변 재생성 노드 실행")
    retry = state.get("retry_count", 0) + 1
    effective_query = state.get("rewritten_query") or state["question"]
    results = search_documents(
        query=effective_query,
        top_k=state.get("top_k", 4) + 2,
        document_id=state.get("document_id"),
    )
    context = "\n---\n".join([
        f"[출처: {r['document_id']}, 페이지: {r.get('page', '?')}]\n{r['text']}"
        for r in results
    ]) if results else ""
    answer = generate_rag_answer(
        question=state["question"], context=context,
        chat_history=state.get("chat_history", []),
    ) if context else "죄송합니다. 관련 정보를 충분히 찾지 못했습니다."
    trace = state.get("node_trace", []) + ["regenerate"]
    return {
        "answer": answer, "context": context, "sources": results,
        "retry_count": retry, "node_trace": trace,
    }
 
 
# 조건부 엣지 함수
def should_rewrite_query(state: RAGState) -> str:
    if state.get("use_query_rewrite", True):
        return "rewrite"
    return "skip_rewrite"
 
 
def should_filter(state: RAGState) -> str:
    if state.get("use_rerank", False) and state.get("search_results"):
        return "filter"
    return "skip_filter"
 
 
def should_retry(state: RAGState) -> str:
    scores = state.get("grade_scores", {})
    retry_count = state.get("retry_count", 0)
    max_retries = settings.LANGGRAPH_MAX_RETRIES
    faithfulness = scores.get("faithfulness", 0.0)
    hallucination = scores.get("hallucination", 1.0)
    if retry_count < max_retries and (faithfulness < 0.5 or hallucination > 0.5):
        logger.info(f"[LangGraph] 품질 부적합, 재생성 (시도 {retry_count + 1}/{max_retries})")
        return "retry"
    return "accept"
 
 
# 그래프 구성
def build_rag_graph() -> StateGraph:
    workflow = StateGraph(RAGState)
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("query_routing", query_routing_node)
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("relevance_filter", relevance_filter_node)
    workflow.add_node("build_context", build_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("grade_answer", grade_answer_node)
    workflow.add_node("regenerate", regenerate_node)
 
    workflow.set_entry_point("initialize")
    workflow.add_conditional_edges(
        "initialize", should_rewrite_query,
        {"rewrite": "query_rewrite", "skip_rewrite": "query_routing"},
    )
    workflow.add_edge("query_rewrite", "query_routing")
    workflow.add_edge("query_routing", "vector_search")
    workflow.add_conditional_edges(
        "vector_search", should_filter,
        {"filter": "relevance_filter", "skip_filter": "build_context"},
    )
    workflow.add_edge("relevance_filter", "build_context")
    workflow.add_edge("build_context", "generate_answer")
    workflow.add_edge("generate_answer", "grade_answer")
    workflow.add_conditional_edges(
        "grade_answer", should_retry,
        {"retry": "regenerate", "accept": END},
    )
    workflow.add_edge("regenerate", END)
    return workflow
 
 
_compiled_graph = build_rag_graph().compile()
 
 
@traceable(name="rag_workflow", run_type="chain")
def run_rag_workflow(
    question: str,
    session_id: str,
    document_id: Optional[str] = None,
    top_k: int = 4,
    use_rerank: bool = False,
    use_query_rewrite: bool = True,
) -> Dict:
    """LangGraph RAG 워크플로우 실행"""
    logger.info(f"[LangGraph] 워크플로우 시작 - 질문: {question[:50]}...")
    initial_state: RAGState = {
        "question": question, "session_id": session_id,
        "document_id": document_id, "top_k": top_k,
        "use_rerank": use_rerank, "use_query_rewrite": use_query_rewrite,
        "rewritten_query": None, "route": None,
        "search_results": [], "filtered_results": [],
        "context": "", "chat_history": [],
        "answer": "", "sources": [], "grade_scores": {},
        "node_trace": [], "retry_count": 0,
    }
    final_state = _compiled_graph.invoke(initial_state)
    logger.info(f"[LangGraph] 워크플로우 완료 - 노드 추적: {final_state.get('node_trace')}")
    return {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "rewritten_query": final_state.get("rewritten_query"),
        "route": final_state.get("route"),
        "node_trace": final_state.get("node_trace", []),
        "grade_scores": final_state.get("grade_scores", {}),
    }
 

#  [추가] LangSmith 평가 (기존 langsmith_eval.py 통합)

_langsmith_client = None
 
 
def get_langsmith_client():
    global _langsmith_client
    if _langsmith_client is None:
        if not settings.LANGSMITH_API_KEY:
            logger.warning("LANGSMITH_API_KEY가 설정되지 않았습니다.")
            return None
        try:
            from langsmith import Client
            _langsmith_client = Client(
                api_key=settings.LANGSMITH_API_KEY,
                api_url=settings.LANGSMITH_ENDPOINT,
            )
            logger.info(f"LangSmith 클라이언트 초기화 (프로젝트: {settings.LANGSMITH_PROJECT})")
        except Exception as e:
            logger.error(f"LangSmith 클라이언트 초기화 실패: {e}")
            return None
    return _langsmith_client
 
 
def create_eval_dataset(
    dataset_name: str, examples: List[Dict[str, Any]], description: str = "",
) -> Optional[str]:
    client = get_langsmith_client()
    if not client:
        return None
    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=description or f"RAG 평가 데이터셋 - {datetime.now().isoformat()}",
        )
        for ex in examples:
            client.create_example(
                inputs={"question": ex["question"]},
                outputs={
                    "expected_answer": ex.get("expected_answer", ""),
                    "context": ex.get("context", ""),
                },
                dataset_id=dataset.id,
            )
        logger.info(f"평가 데이터셋 생성 완료: {dataset_name} ({len(examples)}개 예제)")
        return str(dataset.id)
    except Exception as e:
        logger.error(f"평가 데이터셋 생성 실패: {e}")
        return None
 
 
def list_eval_datasets() -> List[Dict]:
    client = get_langsmith_client()
    if not client:
        return []
    try:
        datasets = list(client.list_datasets())
        return [
            {
                "id": str(ds.id), "name": ds.name,
                "description": ds.description,
                "created_at": ds.created_at.isoformat() if ds.created_at else None,
                "example_count": ds.example_count,
            }
            for ds in datasets
        ]
    except Exception as e:
        logger.error(f"데이터셋 목록 조회 실패: {e}")
        return []
 
 
def faithfulness_evaluator(run, example) -> dict:
    try:
        predicted = run.outputs.get("answer", "")
        question = run.inputs.get("question", "")
        context = run.outputs.get("context", "")
        if not context:
            return {"key": "faithfulness", "score": 0.0}
        scores = grade_answer(question=question, context=context, answer=predicted)
        return {"key": "faithfulness", "score": scores.get("faithfulness", 0.0)}
    except Exception as e:
        logger.error(f"충실성 평가 실패: {e}")
        return {"key": "faithfulness", "score": 0.0}
 
 
def relevance_evaluator(run, example) -> dict:
    try:
        predicted = run.outputs.get("answer", "")
        question = run.inputs.get("question", "")
        context = run.outputs.get("context", "")
        if not predicted:
            return {"key": "relevance", "score": 0.0}
        scores = grade_answer(question=question, context=context or "", answer=predicted)
        return {"key": "relevance", "score": scores.get("relevance", 0.0)}
    except Exception as e:
        logger.error(f"관련성 평가 실패: {e}")
        return {"key": "relevance", "score": 0.0}
 
 
def answer_correctness_evaluator(run, example) -> dict:
    try:
        predicted = run.outputs.get("answer", "")
        expected = example.outputs.get("expected_answer", "")
        if not expected:
            return {"key": "correctness", "score": None, "comment": "기대 답변 없음"}
        predicted_lower = predicted.lower()
        expected_words = set(expected.lower().split())
        matched = sum(1 for w in expected_words if w in predicted_lower)
        score = matched / len(expected_words) if expected_words else 0.0
        return {"key": "correctness", "score": min(score, 1.0)}
    except Exception as e:
        logger.error(f"정확성 평가 실패: {e}")
        return {"key": "correctness", "score": 0.0}
 
 
def run_evaluation(
    dataset_name: str, experiment_prefix: str = "rag-eval",
    evaluators: Optional[List[Callable]] = None,
) -> Optional[Dict]:
    client = get_langsmith_client()
    if not client:
        return None
    try:
        from langsmith.evaluation import evaluate
 
        def target(inputs: dict) -> dict:
            result = run_rag_workflow(
                question=inputs["question"], session_id="eval-session",
                document_id=inputs.get("document_id"),
            )
            return {
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "context": "\n".join([s["text"] for s in result.get("sources", [])]),
            }
 
        if evaluators is None:
            evaluators = [faithfulness_evaluator, relevance_evaluator, answer_correctness_evaluator]
        results = evaluate(
            target, data=dataset_name, evaluators=evaluators,
            experiment_prefix=experiment_prefix, client=client,
        )
        logger.info(f"평가 완료: {experiment_prefix}")
        return {"experiment_prefix": experiment_prefix, "dataset_name": dataset_name, "status": "completed"}
    except Exception as e:
        logger.error(f"평가 실행 실패: {e}")
        return {"status": "failed", "error": str(e)}
 
 
def log_feedback(run_id: str, key: str, score: float, comment: str = "") -> bool:
    client = get_langsmith_client()
    if not client:
        return False
    try:
        client.create_feedback(run_id=run_id, key=key, score=score, comment=comment)
        logger.info(f"피드백 기록 완료: run={run_id}, {key}={score}")
        return True
    except Exception as e:
        logger.error(f"피드백 기록 실패: {e}")
        return False