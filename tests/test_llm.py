from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services import llm as llm_module
from app.services.llm import get_llm, LLMService, RAG_SYSTEM_PROMPT, build_rag_messages


def test_get_llm_use_retry_false_returns_bare_model(monkeypatch):
    # use_retry=False면 재시도 래퍼 없이 원본 모델 인스턴스를 그대로 반환해야 한다.
    fake_instance = MagicMock(name="ChatUpstageInstance")
    monkeypatch.setattr("langchain_upstage.ChatUpstage", MagicMock(return_value=fake_instance))

    result = get_llm(provider="upstage", use_retry=False)

    assert result is fake_instance
    fake_instance.with_retry.assert_not_called()


def test_get_llm_without_retry_supports_structured_output(monkeypatch):
    # 실제 모델 클래스로 검증한다. MagicMock은 존재하지 않는 메서드 접근도 성공하므로
    # "반환 객체에 with_structured_output이 실제로 있는가"는 실물로만 확인할 수 있다.
    # (graph_store.extract_graph_from_text가 이 메서드에 의존한다. 조합 시점엔 네트워크 호출 없음.)
    monkeypatch.setattr(llm_module.settings, "UPSTAGE_API_KEY", "dummy-key-for-test")

    llm = get_llm(provider="upstage", use_retry=False)

    assert callable(getattr(llm, "with_structured_output", None))


def test_retry_wrapped_llm_loses_structured_output(monkeypatch):
    """카나리아: RunnableRetry는 BaseChatModel의 with_structured_output을 위임하지 않는다.

    이 제약 때문에 graph_store.py는 get_llm(use_retry=False)로 원본 모델을 받아
    structured output을 먼저 조합한 뒤 apply_llm_retry를 씌운다.

    이 테스트가 깨진다면 langchain-core가 속성 위임을 추가했다는 신호다.
    그 경우 use_retry=False 우회는 더 이상 필수가 아니므로, 이 테스트를 지우고
    graph_store의 조합 순서를 단순화할 수 있는지 검토하라.
    """
    monkeypatch.setattr(llm_module.settings, "UPSTAGE_API_KEY", "dummy-key-for-test")

    wrapped = get_llm(provider="upstage")  # 기본값 use_retry=True

    assert not hasattr(wrapped, "with_structured_output")


def test_get_llm_unsupported_provider_raises():
    with pytest.raises(ValueError):
        get_llm(provider="unsupported-provider")


def test_get_llm_dispatches_to_upstage_with_settings(monkeypatch):
    fake_instance = MagicMock(name="ChatUpstageInstance")
    fake_chat_upstage = MagicMock(return_value=fake_instance)
    monkeypatch.setattr("langchain_upstage.ChatUpstage", fake_chat_upstage)

    result = get_llm(provider="upstage", temperature=0.5)

    # get_llm은 일시적 오류(429/503 등) 자동 재시도를 위해 with_retry()로 감싼 래퍼를 반환한다.
    assert result is fake_instance.with_retry.return_value
    _, kwargs = fake_chat_upstage.call_args
    assert kwargs["temperature"] == 0.5


def test_get_llm_dispatches_to_openai(monkeypatch):
    fake_instance = MagicMock(name="ChatOpenAIInstance")
    fake_chat_openai = MagicMock(return_value=fake_instance)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", fake_chat_openai)

    result = get_llm(provider="openai", temperature=0.7)

    assert result is fake_instance.with_retry.return_value
    _, kwargs = fake_chat_openai.call_args
    assert kwargs["temperature"] == 0.7


def test_get_llm_dispatches_to_ollama(monkeypatch):
    fake_instance = MagicMock(name="ChatOllamaInstance")
    fake_chat_ollama = MagicMock(return_value=fake_instance)
    monkeypatch.setattr("langchain_community.chat_models.ChatOllama", fake_chat_ollama)

    result = get_llm(provider="ollama", temperature=0.0)

    assert result is fake_instance.with_retry.return_value
    _, kwargs = fake_chat_ollama.call_args
    assert kwargs["temperature"] == 0.0


def test_get_llm_dispatches_to_gemini(monkeypatch):
    fake_instance = MagicMock(name="ChatGoogleGenerativeAIInstance")
    fake_chat_gemini = MagicMock(return_value=fake_instance)
    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", fake_chat_gemini)

    result = get_llm(provider="gemini", temperature=0.1)

    assert result is fake_instance.with_retry.return_value
    _, kwargs = fake_chat_gemini.call_args
    assert kwargs["temperature"] == 0.1


def test_get_llm_dispatches_to_anthropic(monkeypatch):
    fake_instance = MagicMock(name="ChatAnthropicInstance")
    fake_chat_anthropic = MagicMock(return_value=fake_instance)
    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", fake_chat_anthropic)

    result = get_llm(provider="anthropic", temperature=0.3)

    assert result is fake_instance.with_retry.return_value
    _, kwargs = fake_chat_anthropic.call_args
    assert kwargs["temperature"] == 0.3


def test_llm_service_uses_shared_rag_system_prompt(monkeypatch):
    # LLMService.__init__이 실제 LLM 클라이언트를 생성하지 않도록 get_llm을 대체한다.
    monkeypatch.setattr(llm_module, "get_llm", lambda provider=None, temperature=0.2: MagicMock())

    service = LLMService()
    service.model.invoke = MagicMock(return_value=SimpleNamespace(content="테스트 답변", text="테스트 답변"))

    answer = service.generate_answer(question="질문", context="문서 내용")

    assert answer == "테스트 답변"
    sent_messages = service.model.invoke.call_args[0][0]
    # generate_answer가 RAG_SYSTEM_PROMPT를 그대로 재사용하는지 확인한다 (llm.py 내부 중복 프롬프트 제거 검증).
    assert sent_messages[0].content == RAG_SYSTEM_PROMPT
    assert "질문: 질문" in sent_messages[-1].content


def test_llm_service_converts_chat_history_to_messages(monkeypatch):
    from langchain_core.messages import HumanMessage, AIMessage

    monkeypatch.setattr(llm_module, "get_llm", lambda provider=None, temperature=0.2: MagicMock())

    service = LLMService()
    service.model.invoke = MagicMock(return_value=SimpleNamespace(content="답변", text="답변"))

    chat_history = [
        {"role": "user", "content": "이전 질문"},
        {"role": "assistant", "content": "이전 답변"},
    ]
    service.generate_answer(question="새 질문", context="문서", chat_history=chat_history)

    sent_messages = service.model.invoke.call_args[0][0]
    # [system, 이전 user, 이전 assistant, 현재 user] 순서로 구성되어야 한다.
    assert isinstance(sent_messages[1], HumanMessage) and sent_messages[1].content == "이전 질문"
    assert isinstance(sent_messages[2], AIMessage) and sent_messages[2].content == "이전 답변"


def test_build_rag_messages_treats_any_non_user_role_as_assistant():
    from langchain_core.messages import HumanMessage, AIMessage

    # chat.py 스트리밍 엔드포인트가 이 함수를 그대로 재사용하므로, "assistant"로 정확히
    # 일치하지 않는 role(예: 과거의 다른 표기)도 assistant로 취급되는지 확인한다.
    chat_history = [
        {"role": "user", "content": "질문1"},
        {"role": "bot", "content": "답변1"},
    ]
    messages = build_rag_messages("질문2", "컨텍스트", chat_history)

    assert isinstance(messages[1], HumanMessage) and messages[1].content == "질문1"
    assert isinstance(messages[2], AIMessage) and messages[2].content == "답변1"


def test_build_rag_messages_without_history():
    messages = build_rag_messages("질문", "컨텍스트")
    assert messages[0].content == RAG_SYSTEM_PROMPT
    assert "질문: 질문" in messages[-1].content
    assert len(messages) == 2


def test_llm_service_propagates_model_errors(monkeypatch):
    monkeypatch.setattr(llm_module, "get_llm", lambda provider=None, temperature=0.2: MagicMock())

    service = LLMService()
    service.model.invoke = MagicMock(side_effect=RuntimeError("LLM API 오류"))

    with pytest.raises(RuntimeError):
        service.generate_answer(question="질문", context="문서")
