from typing import List, Dict, Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.core.config import settings
from app.utils.logger import logger

def get_llm(provider: Optional[str] = None, temperature: float = 0.2) -> BaseChatModel:
    """LLM 팩토리 함수: LangChain BaseChatModel 인스턴스 반환"""
    provider = provider or settings.LLM_PROVIDER
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.OPENAI_CHAT_MODEL,
                temperature=temperature
            )
        elif provider == "upstage":
            from langchain_upstage import ChatUpstage
            return ChatUpstage(
                api_key=settings.UPSTAGE_API_KEY,
                model=settings.UPSTAGE_CHAT_MODEL,
                temperature=temperature
            )
        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                google_api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_CHAT_MODEL,
                temperature=temperature
            )
        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                temperature=temperature
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                api_key=settings.ANTHROPIC_API_KEY,
                model_name=settings.ANTHROPIC_CHAT_MODEL,
                temperature=temperature
            )
        else:
            raise ValueError(f"지원하지 않는 LLM 공급자: {provider}")
    except Exception as e:
        logger.error(f"LLM 초기화 실패 ({provider}): {e}")
        raise

class LLMService:
    """하위 호환성을 위한 LLM 호출 서비스 래퍼"""
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = get_llm(self.provider)
        logger.info(f"LLM 초기화 완료: {self.provider}")

    def generate_answer(self, question: str, context: str, 
                       chat_history: List[Dict] = None) -> str:
        """문맥과 질문을 바탕으로 답변 생성"""
        try:
            system_prompt = """당신은 제공된 문서를 기반으로 답변하는 도우미입니다.

중요한 규칙:
1. 제공된 문서에서만 정보를 가져와 답변하세요
2. 문서에 없는 내용은 추측하지 마세요
3. 답변할 수 없으면 "문서에서 찾을 수 없습니다"라고 말하세요
4. 항상 사실에 기반한 답변을 제공하세요"""
            messages = [SystemMessage(content=system_prompt)]
            
            if chat_history:
                for msg in chat_history:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))
            
            user_message = f"참고 문서:\n{context}\n\n질문: {question}"
            messages.append(HumanMessage(content=user_message))
            
            response = self.model.invoke(messages)
            logger.debug("LLM 답변 생성 완료")
            return response.content
        except Exception as e:
            logger.error(f"LLM 답변 생성 오류: {str(e)}")
            raise

# LangChain 프롬프트 템플릿
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
 
RAG_QA_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template(
        "참고 문서:\n{context}\n\n질문: {question}"
    ),
])
 
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

# LangChainLLMWrapper 클래스는 더이상 불필요하므로 제거됨.