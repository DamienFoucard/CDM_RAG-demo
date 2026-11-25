"""Checks for the search-before-routing experiment, without model calls."""
import json
from pathlib import Path

import pytest

from app.decision_router import ROUTE_PROMPT
from scripts.benchmark_evidence_router import (
    HIT_LIMIT,
    SNIPPET_CHARS,
    evidence_messages,
    initial_hits,
    initial_query,
    run_case,
    validate_baseline,
)


ROOT = Path(__file__).resolve().parents[1]


def test_raw_query_keeps_prior_messages_and_the_current_question():
    case = {"question": "Which field?", "context_messages": [
        {"role": "user", "content": "Look at Contact."},
        {"role": "assistant", "content": "In applicationCommon."},
    ]}
    assert initial_query(case) == "User: Look at Contact.\nAssistant: In applicationCommon.\nUser: Which field?"
    assert initial_query({"question": "Standalone?", "context_messages": []}) == "Standalone?"


def test_bounded_retrieved_text_is_user_data_and_gold_never_enters_prompt():
    case = {"question": "Which field?", "context_messages": [],
            "expected_route": "rewrite_search", "rationale": "GOLD_DO_NOT_SEND"}
    passages = [{"id": f"p{n}", "title": "Title", "text": "x" * (SNIPPET_CHARS + 90),
                 "uri": "https://example.org", "score": 1.0 / n} for n in range(1, 6)]
    hits = initial_hits(passages)
    assert len(hits) == HIT_LIMIT
    assert [h["rank"] for h in hits] == [1, 2, 3]
    assert all(len(h["text"]) == SNIPPET_CHARS for h in hits)
    messages = evidence_messages(case, hits)
    assert messages[0]["role"] == "system" and messages[0]["content"].startswith(ROUTE_PROMPT)
    assert "p1" not in messages[0]["content"] and "GOLD_DO_NOT_SEND" not in str(messages)
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].endswith("Current user request:\nWhich field?")
    payload = messages[-1]["content"].split("\n", 1)[1].split("\n\nCurrent user request:", 1)[0]
    shown = json.loads(payload)["retrieved_schema_excerpts"]
    assert shown == [{key: hit[key] for key in ("rank", "id", "title", "text")} for hit in hits]


def test_search_precedes_classifier_and_preserves_binary_no_search():
    events = []

    class Index:
        def search(self, query, *, limit, mode):
            events.append(("search", query, limit, mode))
            return [{"id": "irrelevant", "title": "A", "text": "unrelated excerpt", "score": .2}]

    class Model:
        router_model = "router"
        embedding_model = "embedder"

        def chat_json_timed(self, name, messages, **kwargs):
            events.append(("classify", name, messages))
            return {"route": "no_search"}, {"eval_count": 3}

        def runtime_info(self, name):
            return {"model": name}

    case = {"id": "t01", "question": "What is my private balance?", "context_messages": [],
            "accepted_routes": ["no_search"], "source_expected": "SECRET_SCOPE_VALUE"}
    baseline = {"predicted_route": "no_search", "status": "scored", "correct": True, "elapsed_seconds": .2}
    result = run_case(case, Model(), Index(), baseline)
    assert [event[0] for event in events] == ["search", "classify"]
    assert events[0][1:] == (case["question"], 3, "hybrid")
    assert result["status"] == "scored" and result["correct"] is True
    assert result["predicted_route"] == result["baseline_route"] == "no_search"
    assert result["initial_hits"][0]["id"] == "irrelevant"
    assert result["end_to_end_seconds"] >= 0 and result["search_seconds"] >= 0
    assert "SECRET_SCOPE_VALUE" not in str(events[1][2])


def test_benchmark_uses_same_prompt_and_snippet_builders_as_serving():
    from app import decision_router
    from scripts import benchmark_evidence_router

    assert benchmark_evidence_router.evidence_messages is decision_router.evidence_messages
    assert benchmark_evidence_router.initial_hits is decision_router.initial_hits
    assert benchmark_evidence_router.initial_query is decision_router.initial_query


def test_search_failure_is_not_misreported_as_a_router_decision():
    class Index:
        def search(self, *args, **kwargs):
            raise RuntimeError("index unavailable")

    class Model:
        router_model = "router"
        embedding_model = "embedder"

        def chat_json_timed(self, *args, **kwargs):
            raise AssertionError("The classifier must not run when the search fails")

    case = {"id": "t02", "question": "A question", "context_messages": [],
            "expected_route": "raw_search"}
    row = run_case(case, Model(), Index(), {})
    assert row["status"] == "error"
    assert row["predicted_route"] is None and row["correct"] is False
    assert "index unavailable" in row["error"]


def test_saved_baseline_checks_questions_gold_and_prompt_without_model_calls():
    dataset_path = ROOT / "evaluation/routing-decision-v1.json"
    baseline_path = ROOT / "data/macstudio-runs/routing-decisions-test-small.json"
    if not baseline_path.exists():
        pytest.skip("Local saved benchmark is not part of the repository")
    raw = dataset_path.read_bytes()
    data = json.loads(raw)
    baseline = json.loads(baseline_path.read_text())
    rows = validate_baseline(data, raw, baseline, ["qwen3:4b"])
    assert len(rows) == 18
    baseline["rows"][0]["question"] = "tampered"
    with pytest.raises(ValueError, match="differs from its dataset"):
        validate_baseline(data, raw, baseline, [baseline["rows"][0]["model"]])

