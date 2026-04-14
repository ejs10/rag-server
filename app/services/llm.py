from typing import List, Dict, Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.core.config import settings
from app.utils.logger import logger

class LLMService:
    """LLM 호출 서비스 (OpenAI, Ollama, Gemini)"""
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = None
        self.client = None
        self._initialize()

    def _initialize(self):
        """LLM 클라이언트 초기화"""
        try:
            if self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
                self.model = settings.OPENAI_CHAT_MODEL

            elif self.provider == "upstage":
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=settings.UPSTAGE_API_KEY,
                    base_url="https://api.upstage.ai/v1"
                )
                self.model = settings.UPSTAGE_CHAT_MODEL  # 업스테이지 모델
                logger.info(f"Upstage(Solar) LLM 초기화: {self.model}")

            elif self.provider == "gemini":
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=settings.GEMINI_API_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                self.model = settings.GEMINI_CHAT_MODEL  # 빠르고 강력한 무료 모델
                logger.info(f"Gemini LLM 초기화: {self.model}")

            elif self.provider == "ollama":
                from openai import OpenAI
                self.client = OpenAI(
                    base_url=settings.OLLAMA_BASE_URL + "/v1",
                    api_key="ollama"
                )
                self.model = settings.OLLAMA_MODEL
            else:
                raise ValueError(f"지원하지 않는 LLM 공급자: {self.provider}")
        except Exception as e:
            raise

    def generate_answer(self, question: str, context: str, 
                       chat_history: List[Dict] = None) -> str:
        """
        문맥과 질문을 바탕으로 답변 생성
        
        Args:
            question: 사용자 질문
            context: 검색된 문서 내용
            chat_history: 대화 히스토리 (선택사항)
        
        Returns:
            LLM 생성 답변
        """
        try:
            # 시스템 프롬프트: RAG 원칙 강조
            system_prompt = """당신은 제공된 문서를 기반으로 답변하는 도우미입니다.

중요한 규칙:
1. 제공된 문서에서만 정보를 가져와 답변하세요
2. 문서에 없는 내용은 추측하지 마세요
3. 답변할 수 없으면 "문서에서 찾을 수 없습니다"라고 말하세요
4. 항상 사실에 기반한 답변을 제공하세요"""
            messages = [{"role": "system", "content": system_prompt}]
            if chat_history:
                for msg in chat_history:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # 현재 질문 (문맥 포함)
            user_message = f"""참고 문서:
{context}

질문: {question}"""
            messages.append({"role": "user", "content": user_message})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=0.2
            )

            answer = response.choices[0].message.content
            logger.info(f"LLM 답변 생성 완료")
            return answer
        except Exception as e:
            logger.error(f"LLM 답변 생성 오류: {str(e)}")
            raise

# [추가] LangChain 프롬프트 템플릿 (기존 langchain_prompts.py 통합)
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


# [추가] LangChain ChatModel 래퍼 클래스
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
 
 
class LangChainLLMWrapper(BaseChatModel):
    """
    기존 LLMService를 LangChain BaseChatModel 인터페이스로 래핑.
    """
 
    model_config = {"arbitrary_types_allowed": True}  # [수정] Pydantic v2 호환
 
    llm_service: Any  # [수정] = None 제거 (Pydantic v2에서 None 기본값 문제 방지)
 
    def __init__(self, llm_service: LLMService, **kwargs):
        super().__init__(llm_service=llm_service, **kwargs)
 
    @property
    def _llm_type(self) -> str:
        return f"custom-{self.llm_service.provider}"
 
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """LangChain 메시지 형식으로 LLM 호출"""
        openai_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
            else:
                openai_messages.append({"role": "user", "content": msg.content})
 
        response = self.llm_service.client.chat.completions.create(
            model=self.llm_service.model,
            messages=openai_messages,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.2),
        )
 
        content = response.choices[0].message.content
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])