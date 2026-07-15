from typing import List, Dict, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.core.config import settings
from app.utils.logger import logger


def apply_llm_retry(runnable):
    """
    LLM 호출 재시도 정책을 한 곳에서 관리한다.
    429(rate limit), 503(UNAVAILABLE) 등 일시적 오류는 지수 백오프로 최대 3회 자동 재시도한다.
    잘못된 API 키 같은 영구적 오류는 재시도해도 어차피 실패하므로 그대로 재발생한다.
    """
    return runnable.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)


def get_llm(provider: Optional[str] = None, temperature: float = 0.2,
            use_retry: bool = True) -> BaseChatModel:
    """
    설정에 따라 적절한 LLM 인스턴스를 생성해 반환하는 팩토리 함수.
    LangChain의 공통 인터페이스를 반환하므로, LLM_PROVIDER를 바꿔도 호출 코드를 수정할 필요가 없다.

    Parameters:
        provider   : LLM 공급자 ("openai", "upstage", "gemini", "ollama", "anthropic").
                     None이면 config의 LLM_PROVIDER를 사용한다.
        temperature: 0.0(결정적) ~ 1.0(창의적). RAG는 사실 기반 답변이 중요하므로 기본값 0.2.
        use_retry  : True면 apply_llm_retry로 감싼 RunnableRetry를 반환한다.
                     RunnableRetry는 with_structured_output 같은 BaseChatModel 전용 메서드를
                     위임하지 않으므로, 그런 메서드가 필요한 호출부는 False로 원본 모델을 받아
                     조합을 마친 뒤 apply_llm_retry를 직접 적용해야 한다.

    Returns:
        LangChain BaseChatModel 인스턴스 (use_retry=True면 재시도 래퍼가 씌워진 Runnable)
    """
    provider = provider or settings.LLM_PROVIDER
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.OPENAI_CHAT_MODEL,
                temperature=temperature
            )
        elif provider == "upstage":
            from langchain_upstage import ChatUpstage
            llm = ChatUpstage(
                api_key=settings.UPSTAGE_API_KEY,
                model=settings.UPSTAGE_CHAT_MODEL,
                temperature=temperature
            )
        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                google_api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_CHAT_MODEL,
                temperature=temperature
            )
        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            llm = ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                temperature=temperature
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                api_key=settings.ANTHROPIC_API_KEY,
                model_name=settings.ANTHROPIC_CHAT_MODEL,
                temperature=temperature
            )
        else:
            raise ValueError(f"지원하지 않는 LLM 공급자: {provider}")

        if use_retry:
            return apply_llm_retry(llm)
        return llm
    except Exception as e:
        logger.error(f"LLM 초기화 실패 ({provider}): {e}")
        raise


class LLMService:
    """
    chat.py의 기본 /query 엔드포인트에서 사용하는 LLM 호출 래퍼.
    LangGraph 워크플로우에서는 get_llm()을 직접 사용한다.
    """

    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = get_llm(self.provider)
        logger.info(f"LLM 초기화 완료: {self.provider}")

    def generate_answer(self, question: str, context: str,
                        chat_history: List[Dict] = None) -> str:
        """
        검색된 컨텍스트와 질문을 바탕으로 LLM 답변을 생성한다.

        Parameters:
            question    : 사용자 질문
            context     : 벡터 검색으로 찾은 관련 문서 청크들
            chat_history: 이전 대화 목록 [{"role": "user", "content": "..."}]

        Returns:
            LLM이 생성한 답변 문자열
        """
        try:
            messages = build_rag_messages(question, context, chat_history)
            response = self.model.invoke(messages)
            logger.debug("LLM 답변 생성 완료")
            # response.content는 모델에 따라 문자열 또는 콘텐츠 블록 리스트(예: 최신 Gemini의 thinking 응답)일 수 있다.
            # .text는 두 경우 모두 안전하게 문자열로 평탄화해준다.
            return response.text
        except Exception as e:
            logger.error(f"LLM 답변 생성 오류: {str(e)}")
            raise


from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

RAG_SYSTEM_PROMPT = """당신은 제공된 문서를 기반으로 답변하는 도우미입니다.

중요한 규칙:
1. 제공된 문서에서만 정보를 가져와 답변하세요
2. 문서에 없는 내용은 추측하지 마세요
3. 답변할 수 없으면 "문서에서 찾을 수 없습니다"라고 말하세요
4. 항상 사실에 기반한 답변을 제공하세요
5. 답변은 명확하고 구조적으로 작성하세요"""


def build_rag_messages(question: str, context: str, chat_history: Optional[List[Dict]] = None) -> List:
    """
    RAG_SYSTEM_PROMPT + 대화 히스토리 + 현재 질문을 LangChain 메시지 리스트로 조립한다.
    LLMService.generate_answer와 chat.py의 스트리밍 엔드포인트가 동일한 로직을 각자
    구현하다 보면 조용히 어긋날 수 있으므로(예: 히스토리 role 처리 방식), 이 함수 하나로 통일한다.

    Parameters:
        question    : 사용자 질문
        context     : 검색된 참고 문서 텍스트
        chat_history: 이전 대화 목록 [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        [SystemMessage, ...history, HumanMessage] 형태의 LangChain 메시지 리스트
    """
    messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)]
    if chat_history:
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=f"참고 문서:\n{context}\n\n질문: {question}"))
    return messages


RAG_QA_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template(
        "참고 문서:\n{context}\n\n질문: {question}"
    ),
])

# 구어체 질문을 명사 중심의 검색 쿼리로 변환한다. 예: "그거 어디서 샀어?" → "제품 구매처 판매 채널"
QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """당신은 검색 쿼리 최적화 전문가입니다.
사용자의 질문을 벡터 검색에 최적화된 형태로 재작성하세요.

규칙:
1. 핵심 키워드와 의미를 보존하세요
2. 불필요한 조사, 어미를 제거하세요
3. 검색에 효과적인 명사 중심으로 재작성하세요
4. 대화 히스토리가 있으면 맥락을 반영하세요
5. 재작성된 쿼리만 반환하세요 (설명 없이)"""
    ),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template(
        "원본 질문: {question}\n\n재작성된 검색 쿼리:"
    ),
])

QUERY_ROUTING_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """당신은 검색 라우팅 전문가입니다.
사용자의 질문을 분석하여 가장 적합한 검색 방식을 결정하세요.

검색 방식:
- "vector": 일반적인 의미 기반 유사도 검색 (대부분의 질문)
- "graph": 엔티티 간 관계, 연결, 구조를 묻는 질문
- "hybrid": 복합적 질문 (의미 검색 + 관계 검색 모두 필요)

반드시 "vector", "graph", "hybrid" 중 하나만 답변하세요."""
    ),
    HumanMessagePromptTemplate.from_template("질문: {question}"),
])

# Self-RAG: 답변 후 LLM이 스스로 품질을 채점해 미달이면 재시도를 유도한다.
ANSWER_GRADING_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """당신은 RAG 시스템의 답변 품질 평가자입니다.
제공된 컨텍스트와 답변을 비교하여 품질을 평가하세요.

평가 기준:
1. 충실성(faithfulness): 답변이 컨텍스트에 근거하는지 (0.0~1.0)
2. 관련성(relevance): 답변이 질문에 적절히 대답하는지 (0.0~1.0)
3. 환각(hallucination): 컨텍스트에 없는 정보가 포함되었는지 (0.0~1.0, 낮을수록 좋음)

JSON 형식으로만 답변하세요:
{{"faithfulness": 0.0, "relevance": 0.0, "hallucination": 0.0, "explanation": "..."}}"""
    ),
    HumanMessagePromptTemplate.from_template(
        "컨텍스트:\n{context}\n\n질문: {question}\n\n답변: {answer}\n\n평가:"
    ),
])

# "yes"/"no"만 반환하도록 해서 응답 파싱을 단순하게 만든다.
DOCUMENT_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """당신은 문서 관련성 평가자입니다.
주어진 질문에 대해 검색된 문서가 관련이 있는지 판단하세요.

"yes" 또는 "no"로만 답변하세요."""
    ),
    HumanMessagePromptTemplate.from_template(
        "질문: {question}\n\n문서 내용:\n{document}\n\n관련 여부:"
    ),
])
