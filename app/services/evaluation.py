import os
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
from app.core.config import settings
from app.utils.logger import logger
from app.services.rag_pipeline import run_rag_workflow, grade_answer

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
        logger.debug(f"피드백 기록 완료: run={run_id}, {key}={score}")
        return True
    except Exception as e:
        logger.error(f"피드백 기록 실패: {e}")
        return False
