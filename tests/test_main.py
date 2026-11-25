import pytest

from app.main import env_flag


def test_env_flag_uses_stable_judge_default(monkeypatch):
    monkeypatch.delenv("CDM_RAG_JUDGE_ENABLED", raising=False)

    assert env_flag("CDM_RAG_JUDGE_ENABLED", False) is False


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("0", False), ("on", True)])
def test_env_flag_accepts_explicit_values(monkeypatch, value, expected):
    monkeypatch.setenv("CDM_RAG_JUDGE_ENABLED", value)

    assert env_flag("CDM_RAG_JUDGE_ENABLED", False) is expected
