from pathlib import Path

from app.core.config import Settings

ENV_EXAMPLE_PATH = Path(__file__).resolve().parent.parent / ".env.example"


def _parse_env_example_keys() -> set:
    keys = set()
    for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        keys.add(key)
    return keys


def test_env_example_exists():
    assert ENV_EXAMPLE_PATH.exists(), ".env.example 템플릿이 없습니다. 신규 개발자 온보딩을 위해 필요합니다."


def test_env_example_keys_match_settings_fields():
    """
    .env.example의 키 목록과 Settings 필드가 항상 1:1로 일치하는지 검증한다.
    config.py에 새 설정이 추가/삭제될 때 .env.example이 갱신되지 않고 방치되는 것을 방지한다.
    """
    env_keys = _parse_env_example_keys()
    settings_fields = set(Settings.model_fields.keys())

    missing_in_example = settings_fields - env_keys
    stale_in_example = env_keys - settings_fields

    assert not missing_in_example, f".env.example에 누락된 설정: {missing_in_example}"
    assert not stale_in_example, f".env.example에 더 이상 존재하지 않는 설정: {stale_in_example}"
