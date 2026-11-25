import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.evaluation import adaptive_retrieval_benchmark as adaptive
from app.runtime.ollama import ModelError
from app.schema.corpus import load_catalog, render_chunks


class Router:
    def __init__(self, result):
        self.result = result
        self.messages = None

    def chat_json(self, model, messages, **kwargs):
        self.messages = messages
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_expanded_cases_reference_catalog_and_source_revision():
    root = Path(__file__).resolve().parents[1]
    dataset = adaptive.load_adaptive_retrieval_dataset()
    catalog = load_catalog(root / "seed" / "catalog.json")
    passages = {item["id"] for item in render_chunks(catalog)}
    entities = {item["id"] for item in catalog["entities"]}
    assert dataset["source_commit"] == catalog["source_commit"]
    assert len(dataset["questions"]) >= 31
    for case in dataset["questions"]:
        assert set(case.get("positive_entity_ids", [])) <= entities
        assert set(case.get("positive_passage_ids", [])) <= passages


def test_candidate_challenges_reference_current_corpus():
    root = Path(__file__).resolve().parents[1]
    dataset = adaptive.load_adaptive_retrieval_dataset(adaptive.CHALLENGE_DATASET_PATH)
    catalog = load_catalog(root / "seed" / "catalog.json")
    passages = {item["id"] for item in render_chunks(catalog)}
    assert dataset["source_commit"] == catalog["source_commit"]
    assert {q["split"] for q in dataset["questions"]} == {"development", "test"}
    for case in dataset["questions"]:
        assert len(case["positive_passage_ids"]) >= 2
        assert set(case["positive_passage_ids"]) <= passages


def test_router_receives_only_question_context_and_bounded_passages():
    router = Router({"search_again": True, "query": "find Contact"})
    result = adaptive.route_followup("How related?", [], [{"id": "p1", "text": "x" * 4000,
                                      "positive_entity_ids": ["secret-gold"]}], router, "router")
    assert result == {"search_again": True, "query": "find Contact"}
    prompt = json.loads(router.messages[1]["content"])
    assert len(prompt["first_pass_evidence"][0]["text"]) == 2500
    assert "secret-gold" not in json.dumps(prompt)


@pytest.mark.parametrize("result", [
    {"search_again": True, "query": ""},
    {"search_again": False, "query": "unwanted"},
    {"search_again": "yes", "query": "x"},
    {"search_again": False, "query": "", "other": 1},
])
def test_router_rejects_invalid_or_inconsistent_decisions(result):
    with pytest.raises(ModelError):
        adaptive.route_followup("question", [], [], Router(result), "router")


def test_fusion_deduplicates_and_reinforces_shared_passages():
    first = [{"id": "a"}, {"id": "b"}]
    second = [{"id": "b"}, {"id": "c"}]
    assert [x["id"] for x in adaptive.fuse_passages(first, second)] == ["b", "a", "c"]


def test_scoring_top_twenty_preserves_extra_evidence_and_all_coverage():
    candidates = [{"id": f"p{i}", "entity_id": f"entity{i}"} for i in range(20)]
    score = adaptive._score({"positive_passage_ids": ["p1", "p12"]}, candidates)
    assert score["recall_at_6"] == 0.5
    assert not score["all_at_6"]
    assert score["recall_at_20"] == 1.0
    assert score["all_at_20"]
    assert score["mrr_at_20"] == 0.5


def test_conditional_summary_uses_only_executed_variants():
    score = {"latency_ms": 1, "hit_at_1": 1, "hit_at_3": 1, "hit_at_6": 1,
             "recall_at_6": 1, "mrr_at_6": 1, "recall_at_20": 1, "mrr_at_20": 1,
             "all_at_6": True, "all_at_20": True}
    rows = [{"variants": {"first_pass": score, "followup_only": score}},
            {"variants": {"first_pass": score, "followup_only": None}}]
    assert adaptive._summary(rows, "followup_only")["cases"] == 1
    assert adaptive._summary(rows, "followup_only")["all_at_20"] == 1


def test_human_reference_is_independent_and_only_compared_on_its_own_case(monkeypatch, tmp_path):
    prompts, searches = [], []

    class Model:
        def __init__(self, *args):
            pass
        def chat_json(self, model, messages, **kwargs):
            assert self.chat_context_cap_override == 8192
            prompts.append(messages)
            return {"search_again": False, "query": ""}
        def unload_if_loaded(self, name):
            pass

    class Index:
        catalog = {"source_commit": "abc"}
        index_revision = "rev"
        index_reused = True
        graph = SimpleNamespace(entities={"gold": {}})
        by_id = {}
        def __init__(self, *args):
            pass
        def search(self, query, **kwargs):
            searches.append(query)
            return [{"id": "gold-p", "entity_id": "gold", "text": "evidence"}]

    monkeypatch.setattr(adaptive, "Ollama", Model)
    monkeypatch.setattr(adaptive, "SchemaIndex", Index)
    monkeypatch.setattr(adaptive, "available_embedding_models", lambda _: [{"name": "embed"}])
    monkeypatch.setattr(adaptive, "available_chat_models", lambda _: [{"name": "router"}])
    config = SimpleNamespace(base_url="local", answer_model="answer", scope_model="scope", judge_model="judge",
                             request_timeout=5, stream_idle_timeout=5, first_token_timeout=5)
    dataset = {"dataset_id": "d", "status": "provisional", "source_commit": "abc", "questions": [
        {"id": "q1", "question": "question one", "split": "development",
         "positive_entity_ids": ["gold"], "human_reference_query": "private human wording"},
        {"id": "q2", "question": "question two", "split": "development", "positive_entity_ids": ["gold"]}]}
    result = adaptive.run_adaptive_retrieval_benchmark(
        tmp_path, tmp_path, config, "embed", ["q1", "q2"], "development", dataset, "run", threading.Lock(),
        lambda event: None, lambda record: True, query_model="router")
    metrics = result["summaries"][0]["metrics"]
    assert metrics["first_pass"]["cases"] == 2
    assert metrics["human_reference"]["cases"] == 1
    assert metrics["human_reference_first_pass"]["cases"] == 1
    assert metrics["followup_only"]["cases"] == 0
    assert searches.count("private human wording") == 1
    assert all("private human wording" not in json.dumps(prompt) for prompt in prompts)
    assert all("positive_entity_ids" not in json.dumps(prompt) for prompt in prompts)
    assert result["rows"][1]["variants"]["human_reference"] is None
    assert result["router_context_cap_tokens"] == 8192
    assert result["summaries"][0]["router_context_cap_tokens"] == 8192
    assert result["rows"][0]["router_context_requested_tokens"] == 4096


def test_dataset_loader_rejects_duplicate_ids(tmp_path):
    case = {"id": "same", "question": "what?", "split": "development", "positive_entity_ids": ["a"]}
    dataset = {"dataset_id": "d", "status": "provisional", "source_commit": "abc", "questions": [case, case]}
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset))
    with pytest.raises(ValueError, match="duplicate"):
        adaptive.load_adaptive_retrieval_dataset(path)


def test_default_dataset_loads_extended_cases():
    dataset = adaptive.load_adaptive_retrieval_dataset()
    assert dataset["dataset_id"] == "cdm-adaptive-retrieval-v2"
    assert len(dataset["questions"]) == 31
    assert sum("human_reference_query" in case for case in dataset["questions"]) == 1


@pytest.mark.parametrize("router_result,second_failure,expected_requests,expected_errors", [
    ({"search_again": False, "query": ""}, False, 0, 0),
    (ModelError("bad router"), False, 0, 1),
    ({"search_again": True, "query": "next"}, True, 1, 1),
    ({"search_again": True, "query": "next"}, False, 1, 0),
])
def test_paired_benchmark_fallback_and_counts(monkeypatch, tmp_path, router_result, second_failure,
                                              expected_requests, expected_errors):
    class Model:
        def __init__(self, *args):
            self.router = Router(router_result)
        def chat_json(self, *args, **kwargs):
            return self.router.chat_json(*args, **kwargs)
        def unload_if_loaded(self, name):
            pass

    class Index:
        catalog = {"source_commit": "abc"}
        index_revision = "rev"
        index_reused = True
        graph = SimpleNamespace(entities={"gold": {}})
        by_id = {}
        def __init__(self, *args):
            pass
        def search(self, query, **kwargs):
            if query == "next":
                if second_failure:
                    raise RuntimeError("failed")
                return [{"id": "gold-p", "entity_id": "gold"}]
            return [{"id": "other-p", "entity_id": "other"}, {"id": "gold-p", "entity_id": "gold"}]

    monkeypatch.setattr(adaptive, "Ollama", Model)
    monkeypatch.setattr(adaptive, "SchemaIndex", Index)
    monkeypatch.setattr(adaptive, "available_embedding_models", lambda _: [{"name": "embed"}])
    monkeypatch.setattr(adaptive, "available_chat_models", lambda _: [{"name": "router"}])
    config = SimpleNamespace(base_url="local", answer_model="answer", scope_model="scope", judge_model="judge",
                             request_timeout=5, stream_idle_timeout=5, first_token_timeout=5)
    dataset = {"dataset_id": "d", "status": "provisional", "source_commit": "abc", "questions": [
        {"id": "q", "question": "question", "split": "development", "positive_entity_ids": ["gold"]}]}
    saved, events = [], []
    result = adaptive.run_adaptive_retrieval_benchmark(
        tmp_path, tmp_path, config, "embed", ["q"], "development", dataset, "run", threading.Lock(),
        events.append, lambda record: saved.append(record) or True, query_model="router")
    summary = result["summaries"][0]
    assert summary["followup_requested_count"] == expected_requests
    assert summary["error_count"] == expected_errors
    assert result["log_persisted"] and saved and events[-1]["type"] == "retrieval_benchmark_completed"
    row = result["rows"][0]["variants"]
    if not expected_requests or second_failure:
        assert row["adaptive"]["ranked_entity_ids"] == row["first_pass"]["ranked_entity_ids"]
        assert row["followup_only"] is None
    else:
        assert row["adaptive"]["ranked_entity_ids"][0] == "gold"
        assert row["followup_only"]["ranked_entity_ids"][0] == "gold"
        assert summary["metrics"]["followup_only_first_pass"]["cases"] == 1
    assert row["adaptive"]["latency_ms"] >= row["first_pass"]["latency_ms"]
