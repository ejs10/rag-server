from types import SimpleNamespace
from unittest.mock import MagicMock

from app.services import graph_store as graph_store_module
from app.services.graph_store import GraphStore, KnowledgeGraph


def test_extract_graph_applies_structured_output_before_retry(monkeypatch):
    # 조립 순서 회귀 방지: 원본 모델 → with_structured_output → apply_llm_retry 순서여야 한다.
    # 반대로 재시도 래퍼(RunnableRetry)를 먼저 씌우면 with_structured_output이 없어
    # AttributeError가 나고, 예외가 삼켜져 그래프 추출이 전량 무음 실패한다.
    gs = GraphStore.__new__(GraphStore)
    gs.driver = None  # extract_graph_from_text는 driver를 사용하지 않는다

    bare_llm = MagicMock(name="bare_llm")
    structured = bare_llm.with_structured_output.return_value
    structured.invoke.return_value = KnowledgeGraph()

    fake_get_llm = MagicMock(return_value=bare_llm)
    monkeypatch.setattr(graph_store_module, "get_llm", fake_get_llm)
    retry_spy = MagicMock(side_effect=lambda runnable: runnable)
    monkeypatch.setattr(graph_store_module, "apply_llm_retry", retry_spy)

    result = gs.extract_graph_from_text("아무 텍스트")

    # 원본 모델을 요청했는지 (재시도 래퍼가 아니라)
    assert fake_get_llm.call_args.kwargs.get("use_retry") is False
    # structured output이 원본 모델에 적용됐는지
    bare_llm.with_structured_output.assert_called_once_with(KnowledgeGraph)
    # 재시도는 완성된 structured 파이프라인에 씌워졌는지
    retry_spy.assert_called_once_with(structured)
    assert isinstance(result, KnowledgeGraph)


def _make_graph_store_with_mock_driver():
    """
    GraphStore.__init__은 실제 Neo4j 연결을 시도하므로, __new__로 인스턴스만 만들고
    driver/database를 목으로 직접 채워 넣는다.
    """
    gs = GraphStore.__new__(GraphStore)
    gs.driver = MagicMock()
    gs.database = "neo4j"

    mock_session = MagicMock()
    mock_session.run.return_value = []
    gs.driver.session.return_value.__enter__.return_value = mock_session
    gs.driver.session.return_value.__exit__.return_value = False

    return gs, mock_session


def test_search_graph_skips_empty_keywords(monkeypatch):
    # LLM이 trailing comma나 빈 항목을 포함해 응답해도, Cypher CONTAINS ''가
    # 모든 엔티티에 매칭되지 않도록 빈 키워드는 세션에 전달되지 말아야 한다.
    gs, mock_session = _make_graph_store_with_mock_driver()

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(content="키워드1, , 키워드2,", text="키워드1, , 키워드2,")
    monkeypatch.setattr(graph_store_module, "get_llm", lambda temperature=0.0: fake_llm)

    gs.search_graph("아무 질문")

    called_keywords = [call.kwargs.get("kw") for call in mock_session.run.call_args_list]
    assert "" not in called_keywords
    assert called_keywords == ["키워드1", "키워드2"]


def test_search_graph_returns_empty_when_no_valid_keywords(monkeypatch):
    gs, mock_session = _make_graph_store_with_mock_driver()

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(content=" , , ", text=" , , ")
    monkeypatch.setattr(graph_store_module, "get_llm", lambda temperature=0.0: fake_llm)

    result = gs.search_graph("아무 질문")

    assert result == []
    mock_session.run.assert_not_called()


def test_search_graph_returns_empty_when_driver_missing():
    gs = GraphStore.__new__(GraphStore)
    gs.driver = None

    assert gs.search_graph("아무 질문") == []
