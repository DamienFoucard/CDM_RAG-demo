"""The local decision and query models remain independently selectable."""
from types import SimpleNamespace
import threading

from app import main
from app.decision_router import LocalDecisionRouter
from app.ollama import Ollama


def test_legacy_client_defaults_decision_model_to_query_model():
    models = Ollama("http://localhost", "embed", "query", "answer")
    assert models.decision_model == "query"


def test_decision_and_query_settings_update_independently(monkeypatch):
    models = SimpleNamespace(base_url="http://localhost", embedding_model="embed",
                             answer_model="answer", router_model="query-old",
                             decision_model="decision-old", scope_model="scope", judge_model="judge",
                             unload_if_loaded=lambda _tag: None)
    saved = []
    monkeypatch.setattr(main.app.state, "models", models, raising=False)
    monkeypatch.setattr(main.app.state, "model_run_lock", threading.Lock(), raising=False)
    monkeypatch.setattr(main.app.state, "query_router", LocalDecisionRouter("decision-old", use_evidence=True),
                        raising=False)
    monkeypatch.setattr(main.app.state, "judge_enabled", False, raising=False)
    monkeypatch.setattr(main, "available_chat_models", lambda _url: [
        {"name": "query-new"}, {"name": "decision-new"}])
    monkeypatch.setattr(main, "save_model_roles", lambda roles: saved.append(dict(roles)))

    response = main.update_model_settings(main.ModelSettingsUpdate(router_model="query-new",
                                                                    decision_model="decision-new"))

    assert models.router_model == "query-new"
    assert models.decision_model == "decision-new"
    assert main.app.state.query_router == LocalDecisionRouter("decision-new", use_evidence=True)
    assert saved[0]["router_model"] == "query-new" and saved[0]["decision_model"] == "decision-new"
    assert response["selected"]["decision_model"] == "decision-new"
    assert response["router_backend"] == "ollama-evidence"
