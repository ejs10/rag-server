from typing import List, Dict
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