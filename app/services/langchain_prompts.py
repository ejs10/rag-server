"""
[신규 파일] LangChain 프롬프트 템플릿 정의
경로: app/services/langchain_prompts.py
 
기존 LLMService의 하드코딩된 프롬프트를 LangChain PromptTemplate으로 관리합니다.
"""
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
 

# 1. RAG 답변 생성 프롬프트
 
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
 

# 2. 쿼리 재작성 프롬프트
 
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
 
 
# 3. 쿼리 라우팅 프롬프트 (벡터 / 그래프 / 하이브리드)
 
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
 
 
# ─────────────────────────────────────────────
# 4. 답변 품질 평가 프롬프트 (Self-RAG / Hallucination Check)
# ─────────────────────────────────────────────
 
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
 

# 5. 문서 관련성 평가 프롬프트
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