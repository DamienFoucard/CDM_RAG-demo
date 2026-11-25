"""API integration for the paired adaptive retrieval evaluation."""
import json
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import main


def test_adaptive_stream_dispatches_and_returns_result(monkeypatch):
    dataset = {"dataset_id": "paired-demo", "questions": [{"id": "case-1"}]}
    models = SimpleNamespace(router_model="default-router")
    lock = threading.Lock()
    calls = []
    monkeypatch.setattr(main.app.state, "models", models, raising=False)
    monkeypatch.setattr(main.app.state, "model_run_lock", lock, raising=False)
    monkeypatch.setattr(main, "load_adaptive_retrieval_dataset", lambda: dataset)

    def fake_run(catalog, data_dir, configured_models, embedding_model, question_ids,
                 split, selected_dataset, run_id, model_lock, emit, persist, *, query_model):
        calls.append((configured_models, embedding_model, question_ids, split,
                      selected_dataset, model_lock, query_model))
        emit({"type": "retrieval_benchmark_completed", "result": {"run_id": run_id}})

    monkeypatch.setattr(main, "run_adaptive_retrieval_benchmark", fake_run)
    client = TestClient(main.app, raise_server_exceptions=True)
    response = client.post("/api/v1/retrieval-benchmark/stream", json={
        "track": "adaptive_retrieval", "models": ["embed"], "split": "development",
        "question_ids": ["case-1"], "query_model": "selected-router",
    })

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    event = json.loads(next(line[5:] for line in response.text.splitlines() if line.startswith("data:")))
    assert event["type"] == "retrieval_benchmark_completed"
    assert event["result"]["run_id"]
    assert calls == [(models, "embed", ["case-1"], "development", dataset, lock, "selected-router")]


def test_adaptive_stream_requires_one_embedding_model():
    client = TestClient(main.app, raise_server_exceptions=True)
    for models in ([], ["one", "two"]):
        response = client.post("/api/v1/retrieval-benchmark/stream", json={
            "track": "adaptive_retrieval", "models": models,
            "split": "development", "question_ids": ["case-1"],
        })
        assert response.status_code == 422


def test_candidate_selection_dispatches_candidate_dataset(monkeypatch):
    dataset = {"dataset_id": "candidate", "questions": [{"id": "candidate-1"}]}
    seen = []
    monkeypatch.setattr(main.app.state, "models", SimpleNamespace(router_model="router"), raising=False)
    monkeypatch.setattr(main.app.state, "model_run_lock", threading.Lock(), raising=False)
    def load(path):
        assert path == main.CHALLENGE_DATASET_PATH
        return dataset
    monkeypatch.setattr(main, "load_adaptive_retrieval_dataset", load)
    def run(catalog, data_dir, models, embedding, ids, split, selected, run_id, lock, emit, persist, *, query_model):
        seen.append(selected)
        emit({"type": "retrieval_benchmark_completed", "result": {"run_id": run_id}})
    monkeypatch.setattr(main, "run_adaptive_retrieval_benchmark", run)
    response = TestClient(main.app).post("/api/v1/retrieval-benchmark/stream", json={
        "track": "adaptive_retrieval", "adaptive_question_set": "reformulation_challenge_candidate",
        "models": ["embed"], "split": "development", "question_ids": ["candidate-1"]})
    assert response.status_code == 200
    assert 'retrieval_benchmark_completed' in response.text
    assert seen == [dataset]


def test_candidate_selection_is_not_silently_ignored_for_other_tracks():
    response = TestClient(main.app).post("/api/v1/retrieval-benchmark/stream", json={
        "track": "passages", "adaptive_question_set": "reformulation_challenge_candidate",
        "models": ["embed"], "split": "development", "question_ids": ["case-1"]})
    assert response.status_code == 422


def test_adaptive_stream_rejects_inapplicable_options():
    client = TestClient(main.app, raise_server_exceptions=True)
    for extra in ({"query_models": ["model"]}, {"query_methods": ["rewrite"]},
                  {"query_method": "raw"}, {"rerankers": ["bce"]}):
        response = client.post("/api/v1/retrieval-benchmark/stream", json={
            "track": "adaptive_retrieval", "models": ["embed"], "split": "development",
            "question_ids": ["case-1"], **extra,
        })
        assert response.status_code == 422
