from typing import List, Dict, Optional, Any, Callable, TypedDict
import os
from datetime import datetime
import json
from app.services.document_loader import DocumentLoader
from app.services.vector_store import VectorStore
from app.services.text_splitter import TextSplitter
from app.services.embedding import EmbeddingService, LangChainEmbeddingsWrapper
from app.services.graph_store import GraphStore
from app.core.config import settings
from app.utils.logger import logger
from app.utils.file_handler import generate_document_id


class RAGPipeline:
    """
    문서 처리 전체 파이프라인을 조율하는 오케스트레이터.
    로더·청킹·임베딩·벡터 DB·그래프 DB를 한 곳에서 관리해 처리 순서의 일관성을 보장한다.
    """

    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.graph_store = GraphStore()

        # ChromaDB는 벡터와 텍스트만 저장하므로, 파일명·크기 같은 메타데이터는 별도 JSON에 영속화한다.
        self.metadata_file = os.path.join(settings.VECTOR_DB_PATH, "metadata.json")
        self.documents_metadata = self._load_metadata()

    def _load_metadata(self):
        """서버 재시작 시 이전에 저장된 문서 메타데이터를 복원한다."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"메타데이터 로드 실패: {e}")
        return {}

    def _save_metadata(self):
        """현재 메모리의 메타데이터를 JSON 파일로 저장한다."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                # ensure_ascii=False: 한글을 \uXXXX 이스케이프 없이 그대로 저장
                json.dump(self.documents_metadata, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"메타데이터 저장 실패: {e}")

    def process_document(self, file_path: str, filename: str) -> str:
        """
        문서 전체 처리 파이프라인 (동기 함수).
        HuggingFace와 ChromaDB가 동기 라이브러리이므로, 호출 측에서 run_in_threadpool로 감싸야 한다.

        Parameters:
            file_path: 디스크에 저장된 파일 경로
            filename : 사용자가 업로드한 원본 파일명

        Returns:
            document_id, filename, total_chunks, file_size를 담은 dict
        """
        try:
            text, page_numbers = self.loader.load_document(file_path)
            chunks = self.splitter.split_text(text, page_numbers)

            if not chunks:
                raise ValueError("청크 생성 실패: 생성된 청크가 없습니다")
            logger.debug(f"청크 생성 완료: {len(chunks)} 청크")

            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.embed(chunk_texts)

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
            self._save_metadata()  # 벡터 DB 저장 전에 기록해 서버 장애 시 메타 유실 방지

            self.vector_store.add_documents(
                document_id=document_id,
                embeddings=embeddings.tolist(),
                chunks=chunks
            )

            # 그래프 추출 실패 시에도 벡터 DB 저장은 이미 완료됐으므로 전체 프로세스를 중단하지 않는다.
            kg = self.graph_store.extract_graph_from_text(text)
            if kg and (kg.nodes or kg.relationships):
                self.graph_store.save_graph(kg, document_id)

            logger.debug(f"문서 처리 및 저장 완료: {document_id}")
            return {
                "document_id": document_id,
                "filename": filename,
                "total_chunks": len(chunks),
                "file_size": file_size
            }

        except Exception as e:
            logger.error(f"문서 처리 오류: {str(e)}")
            raise

    def delete_document(self, document_id: str) -> bool:
        """
        업로드 파일, ChromaDB 청크, Neo4j 그래프, 메타데이터 JSON에서 문서를 일괄 삭제한다.

        Returns:
            True: 삭제 성공 / False: 문서가 존재하지 않음
        """
        if document_id not in self.documents_metadata:
            return False
        try:
            meta = self.documents_metadata[document_id]
            file_path = os.path.join(settings.UPLOAD_DIR, meta["filename"])
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"업로드 파일 삭제 완료: {file_path}")

            self.vector_store.delete_document(document_id)
            self.graph_store.delete_graph(document_id)
            del self.documents_metadata[document_id]
            self._save_metadata()
            logger.info(f"문서 삭제 완료: {document_id}")
            return True
        except Exception as e:
            logger.error(f"문서 삭제 오류: {str(e)}")
            raise

    def get_documents_metadata(self) -> List[Dict]:
        """저장된 전체 문서 메타데이터 목록 반환 (메모리에서 읽으므로 빠름)"""
        return list(self.documents_metadata.values())

    def get_document_metadata(self, document_id: str) -> Optional[Dict]:
        """특정 문서 메타데이터 반환"""
        return self.documents_metadata.get(document_id)


# HuggingFace 모델 로드를 포함하므로 요청마다 인스턴스를 새로 만들면 수 초씩 지연된다.
shared_rag_pipeline = RAGPipeline()


from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable

from app.services.llm import (
    get_llm,
    RAG_QA_PROMPT, QUERY_REWRITE_PROMPT, QUERY_ROUTING_PROMPT,
    ANSWER_GRADING_PROMPT, DOCUMENT_RELEVANCE_PROMPT,
)
from app.services.conversation import conversation_manager

_lc_embedding_service = shared_rag_pipeline.embedding_service
langchain_llm = get_llm()
langchain_embeddings = LangChainEmbeddingsWrapper(embedding_service=_lc_embedding_service)


def _chat_history_to_messages(chat_history: Optional[List[Dict]]) -> List:
    """대화 히스토리 [{"role", "content"}]를 LangChain HumanMessage/AIMessage 목록으로 변환한다."""
    messages = []
    if chat_history:
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
    return messages


@traceable(name="query_rewrite")
def rewrite_query(question: str, chat_history: List[Dict] = None) -> str:
    """
    사용자 질문을 벡터 검색에 최적화된 형태로 재작성한다.
    구어체 질문은 임베딩 유사도 검색 성능이 낮으므로 명사 중심으로 변환한다.
    재작성 실패 시 원본 질문으로 폴백해 서비스 중단을 방지한다.

    Parameters:
        question    : 원본 사용자 질문
        chat_history: 이전 대화 목록 (맥락 반영용)

    Returns:
        재작성된 검색 쿼리 문자열
    """
    try:
        messages = _chat_history_to_messages(chat_history)
        chain = QUERY_REWRITE_PROMPT | langchain_llm | StrOutputParser()
        rewritten = chain.invoke({
            "question": question,
            "chat_history": messages if messages else [],
        })
        logger.debug(f"쿼리 재작성: '{question}' → '{rewritten.strip()}'")
        return rewritten.strip()
    except Exception as e:
        logger.warning(f"쿼리 재작성 실패, 원본 사용: {e}")
        return question


@traceable(name="query_routing")
def route_query(question: str) -> str:
    """
    질문 유형에 따라 최적 검색 방식을 결정한다: "vector" / "graph" / "hybrid".
    LANGGRAPH_ROUTING_ENABLED=False이면 항상 config의 SEARCH_MODE를 반환한다.

    Parameters:
        question: 라우팅할 질문 문자열

    Returns:
        "vector" | "graph" | "hybrid"
    """
    if not settings.LANGGRAPH_ROUTING_ENABLED:
        return settings.SEARCH_MODE
    try:
        chain = QUERY_ROUTING_PROMPT | langchain_llm | StrOutputParser()
        route = chain.invoke({"question": question}).strip().lower()
        valid_routes = {"vector", "graph", "hybrid"}
        if route not in valid_routes:
            route = settings.SEARCH_MODE
        logger.debug(f"쿼리 라우팅 결과: {route}")
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
    """
    질문과 의미적으로 가장 유사한 청크를 ChromaDB에서 검색한다.

    Parameters:
        query      : 검색 쿼리 문자열
        top_k      : 반환할 최대 결과 수
        document_id: 지정 시 해당 문서 내에서만 검색

    Returns:
        유사도 순으로 정렬된 청크 딕셔너리 목록
    """
    try:
        query_embedding = _lc_embedding_service.embed_single(query)
        results = shared_rag_pipeline.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
        )
        logger.debug(f"검색 완료: {len(results)}개 청크 반환")
        return results
    except Exception as e:
        logger.error(f"검색 오류: {e}")
        return []


@traceable(name="document_relevance_filter")
def filter_relevant_documents(
    question: str,
    documents: List[Dict],
) -> List[Dict]:
    """
    검색된 문서 중 실제로 질문과 관련 있는 것만 LLM으로 필터링한다.
    벡터 유사도 검색은 의미가 비슷한 것을 반환하지만 질문에 실제로 도움이 되지 않을 수 있다.
    청크마다 LLM을 호출하므로 비용과 지연이 증가한다.

    Parameters:
        question : 사용자 질문
        documents: 필터링할 청크 목록

    Returns:
        관련 있다고 판단된 청크 목록. 전부 걸러지면 원본 목록을 그대로 반환한다.
    """
    if not documents:
        return []
    try:
        chain = DOCUMENT_RELEVANCE_PROMPT | langchain_llm | StrOutputParser()
        filtered = []
        for doc in documents:
            result = chain.invoke({
                "question": question,
                "document": doc["text"][:500],  # 비용 절약을 위해 청크당 500자만 전송
            })
            if "yes" in result.strip().lower():
                filtered.append(doc)
        logger.debug(f"관련성 필터링: {len(documents)}개 → {len(filtered)}개")
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
    """
    검색된 컨텍스트와 질문으로 LangChain 체인을 통해 RAG 답변을 생성한다.

    Parameters:
        question    : 사용자 질문
        context     : 검색된 청크들을 합친 참고 문서 텍스트
        chat_history: 이전 대화 목록

    Returns:
        LLM이 생성한 답변 문자열
    """
    try:
        messages = _chat_history_to_messages(chat_history)
        chain = RAG_QA_PROMPT | langchain_llm | StrOutputParser()
        answer = chain.invoke({
            "question": question,
            "context": context,
            "chat_history": messages if messages else [],
        })
        logger.debug("LangChain RAG 답변 생성 완료")
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
    """
    생성된 답변의 품질을 LLM이 스스로 평가한다 (Self-RAG).

    Parameters:
        question: 사용자 질문
        context : 답변 생성에 사용된 참고 문서
        answer  : 평가할 답변 텍스트

    Returns:
        {"faithfulness": float, "relevance": float, "hallucination": float, "explanation": str}
    """
    try:
        chain = ANSWER_GRADING_PROMPT | langchain_llm | StrOutputParser()
        raw = chain.invoke({
            "question": question,
            "context": context,
            "answer": answer,
        })
        try:
            # LLM이 ```json ... ``` 마크다운 블록으로 감싸 반환하는 경우 벗겨낸다.
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            scores = json.loads(cleaned)
        except json.JSONDecodeError:
            # 파싱 실패 시 최악의 점수로 초기화해 재시도를 유도한다.
            scores = {
                "faithfulness": 0.0, "relevance": 0.0,
                "hallucination": 0.0, "explanation": raw,
            }
        logger.debug(f"답변 품질 평가: {scores}")
        return scores
    except Exception as e:
        logger.warning(f"답변 품질 평가 실패: {e}")
        return {
            "faithfulness": 0.0, "relevance": 0.0,
            "hallucination": 0.0, "explanation": str(e),
        }


from langgraph.graph import StateGraph, END


class RAGState(TypedDict):
    """
    LangGraph 워크플로우 전체에서 공유되는 상태 딕셔너리.
    각 노드 함수는 이 dict를 받아 변경할 키-값 쌍만 담아 반환한다.
    """
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
    node_trace: List[str]  # 실행된 노드 순서 (디버깅용)
    retry_count: int


def initialize_node(state: RAGState) -> dict:
    """초기화 노드: 대화 히스토리 로드 및 이전 실행 잔재 제거."""
    logger.debug("[LangGraph] 초기화 노드 실행")
    chat_history = conversation_manager.get_conversation(state["session_id"])
    return {
        "chat_history": chat_history,
        "node_trace": ["initialize"],
        "retry_count": 0,
        "rewritten_query": None, "route": None,
        "search_results": [], "filtered_results": [], "context": "",
        "answer": "", "sources": [], "grade_scores": {},
    }


def query_rewrite_node(state: RAGState) -> dict:
    """쿼리 재작성 노드: 질문을 검색 최적화 형태로 변환한다."""
    logger.debug("[LangGraph] 쿼리 재작성 노드 실행")
    rewritten = rewrite_query(
        question=state["question"],
        chat_history=state.get("chat_history", []),
    )
    trace = state.get("node_trace", []) + ["query_rewrite"]
    return {"rewritten_query": rewritten, "node_trace": trace}


def query_routing_node(state: RAGState) -> dict:
    """쿼리 라우팅 노드: 검색 방식(vector/graph/hybrid)을 결정한다."""
    logger.debug("[LangGraph] 쿼리 라우팅 노드 실행")
    effective_query = state.get("rewritten_query") or state["question"]
    route = route_query(effective_query)
    trace = state.get("node_trace", []) + ["query_routing"]
    return {"route": route, "node_trace": trace}


def retrieve_node(state: RAGState) -> dict:
    """검색 노드: 결정된 방식으로 관련 청크 또는 그래프 관계를 검색한다."""
    logger.debug("[LangGraph] 지식 검색 노드 실행")
    effective_query = state.get("rewritten_query") or state["question"]
    route = state.get("route", "vector")

    results = []

    if route in ["vector", "hybrid"]:
        vec_results = search_documents(
            query=effective_query,
            top_k=state.get("top_k", 4),
            document_id=state.get("document_id"),
        )
        results.extend(vec_results)

    if route in ["graph", "hybrid"]:
        graph_results = shared_rag_pipeline.graph_store.search_graph(effective_query)
        for gr in graph_results:
            # 그래프 결과를 벡터 결과와 동일한 형식으로 변환해 합친다.
            results.append({
                "document_id": "graph",
                "text": f"[지식 그래프] {gr['source']} -({gr['type']}: {gr['description']})-> {gr['target']}",
                "score": 1.0,
                "chunk_index": -1
            })

    trace = state.get("node_trace", []) + ["retrieve"]
    return {"search_results": results, "node_trace": trace}


def relevance_filter_node(state: RAGState) -> dict:
    """관련성 필터 노드: LLM으로 각 청크의 관련성을 yes/no로 판단한다."""
    logger.debug("[LangGraph] 관련성 필터링 노드 실행")
    effective_query = state.get("rewritten_query") or state["question"]
    filtered = filter_relevant_documents(
        question=effective_query,
        documents=state.get("search_results", []),
    )
    trace = state.get("node_trace", []) + ["relevance_filter"]
    return {"filtered_results": filtered, "node_trace": trace}


def build_context_node(state: RAGState) -> dict:
    """컨텍스트 구성 노드: 청크들을 LLM에 보낼 하나의 텍스트로 합친다."""
    logger.debug("[LangGraph] 컨텍스트 구성 노드 실행")
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
    """답변 생성 노드: 컨텍스트 + 질문 + 히스토리로 LLM 답변을 생성한다."""
    logger.debug("[LangGraph] 답변 생성 노드 실행")
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
    """품질 평가 노드: 생성된 답변의 충실성·관련성·환각 점수를 계산한다."""
    logger.debug("[LangGraph] 답변 품질 평가 노드 실행")
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
    """
    재생성 노드: 품질 미달 시 검색 범위를 넓혀(top_k + 2) 답변을 재생성한다.
    retry_count를 증가시켜 무한 루프를 방지한다.
    """
    logger.debug("[LangGraph] 답변 재생성 노드 실행")
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


def should_rewrite_query(state: RAGState) -> str:
    """쿼리 재작성 여부 결정."""
    if state.get("use_query_rewrite", True):
        return "rewrite"
    return "skip_rewrite"


def should_filter(state: RAGState) -> str:
    """관련성 필터 사용 여부 결정."""
    if state.get("use_rerank", False) and state.get("search_results"):
        return "filter"
    return "skip_filter"


def should_retry(state: RAGState) -> str:
    """
    재시도 여부 결정.
    faithfulness < 0.5(문서 근거 부족) 또는 hallucination > 0.5(거짓 정보 과다)이고
    최대 재시도 횟수 미만이면 재생성한다.
    """
    scores = state.get("grade_scores", {})
    retry_count = state.get("retry_count", 0)
    max_retries = settings.LANGGRAPH_MAX_RETRIES
    faithfulness = scores.get("faithfulness", 0.0)
    hallucination = scores.get("hallucination", 1.0)
    if retry_count < max_retries and (faithfulness < 0.5 or hallucination > 0.5):
        logger.debug(f"[LangGraph] 품질 부적합, 재생성 (시도 {retry_count + 1}/{max_retries})")
        return "retry"
    return "accept"


def build_rag_graph() -> StateGraph:
    """
    LangGraph 워크플로우 그래프를 구성하고 반환한다.

    전체 흐름:
        initialize
          → (조건) query_rewrite 또는 query_routing
          → query_routing → retrieve
          → (조건) relevance_filter 또는 build_context
          → build_context → generate_answer → grade_answer
          → (조건) regenerate 또는 END
    """
    workflow = StateGraph(RAGState)

    workflow.add_node("initialize", initialize_node)
    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("query_routing", query_routing_node)
    workflow.add_node("retrieve", retrieve_node)
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
    workflow.add_edge("query_routing", "retrieve")

    workflow.add_conditional_edges(
        "retrieve", should_filter,
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


# 매 요청마다 그래프를 재빌드하지 않도록 싱글톤으로 컴파일한다.
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
    """
    LangGraph RAG 워크플로우 진입점.
    초기 상태를 컴파일된 그래프에 전달하면 LangGraph가 노드를 순서대로 실행한다.

    Parameters:
        question          : 사용자 질문
        session_id        : 대화 세션 ID (히스토리 연속성 유지)
        document_id       : 검색 범위를 특정 문서로 한정할 때 지정
        top_k             : 검색할 청크 수
        use_rerank        : LLM 기반 관련성 필터 사용 여부
        use_query_rewrite : 쿼리 재작성 사용 여부

    Returns:
        answer, sources, rewritten_query, route, node_trace, grade_scores를 담은 dict
    """
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
