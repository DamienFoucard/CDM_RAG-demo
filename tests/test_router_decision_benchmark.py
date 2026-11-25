"""The isolated router must use the live classifier and never see its labels."""
import pytest

from app.decision_router import ROUTE_SCHEMA, classify_route_timed
from app.ollama import ModelError
from scripts.benchmark_router_decisions import decision_for, messages_for, summarize, validate_dataset


class FakeModel:
    router_model = "router-a"

    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat_json_timed(self, model, messages, **kwargs):
        self.calls.append((model, messages, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response, {"eval_count": 4}

    def runtime_info(self, model):
        return {"model": model}


@pytest.mark.parametrize("route", ["raw_search", "rewrite_search", "no_search"])
def test_route_only_call_keeps_labels_out_and_records_decision(route):
    case = {"id": "case-1", "split": "test", "question": "What does that field mean?",
            "context_messages": [{"role": "assistant", "content": "We are discussing Contact."}],
            "expected_route": route, "rationale": "SECRET REFERENCE", "source_ids": ["SECRET SOURCE"]}
    model = FakeModel({"route": route})
    row = decision_for(case, model)
    assert row["status"] == "scored" and row["correct"] and row["predicted_route"] == route
    assert model.calls[0][0] == "router-a" and model.calls[0][2]["schema"] == ROUTE_SCHEMA
    prompt = repr(model.calls[0][1])
    assert "SECRET REFERENCE" not in prompt and "SECRET SOURCE" not in prompt
    assert model.calls[0][1] == messages_for(case)
    assert not any("query" in message for message in model.calls[0][1])


def test_invalid_route_and_model_error_are_errors_not_false_no_search():
    case = {"id": "x", "question": "Who?", "expected_route": "no_search"}
    for response in ({"route": "decline"}, {"route": "no_search", "query": "invented"}, ModelError("offline")):
        row = decision_for(case, FakeModel(response))
        assert row["status"] == "error" and row["predicted_route"] is None and not row["correct"]


def test_search_gate_separates_subjective_rewrite_from_no_search_errors():
    rows = [
        {"expected_route": "rewrite_search", "predicted_route": "raw_search", "status": "scored", "correct": False, "elapsed_seconds": 1.0},
        {"expected_route": "no_search", "predicted_route": "raw_search", "status": "scored", "correct": False, "elapsed_seconds": 2.0},
        {"expected_route": "raw_search", "predicted_route": None, "status": "error", "correct": False, "elapsed_seconds": 3.0},
    ]
    summary = summarize(rows)
    assert summary["accuracy"] == 0 and summary["search_gate_correct"] == 1
    assert summary["false_search"] == 1 and summary["false_no_search"] == 0 and summary["errors"] == 1
    assert summary["mean_seconds"] == 2.0


def test_binary_scope_accepts_either_search_without_inventing_three_way_label():
    case = {"id": "scope", "question": "Which entity?", "accepted_routes": ["raw_search", "rewrite_search"]}
    first = decision_for(case, FakeModel({"route": "raw_search"}))
    second = decision_for(case, FakeModel({"route": "rewrite_search"}))
    bad = decision_for(case, FakeModel({"route": "no_search"}))
    assert first["correct"] and second["correct"] and not bad["correct"]
    assert first["expected_route"] is None and first["accepted_routes"] == case["accepted_routes"]
    summary = summarize([first, second, bad])
    assert summary["accuracy"] == 2 / 3 and summary["search_gate_correct"] == 2
    assert summary["false_no_search"] == 1 and summary["macro_f1"] is None


def test_dataset_validation_checks_roles_and_uniqueness():
    good = {"dataset_id": "routing", "questions": [{"id": "x", "split": "test", "question": "Hello?",
                                                     "expected_route": "no_search", "context_messages": []}]}
    assert validate_dataset(good) == good["questions"]
    with pytest.raises(ValueError):
        validate_dataset({"dataset_id": "routing", "questions": good["questions"] * 2})
    with pytest.raises(ValueError):
        validate_dataset({"dataset_id": "routing", "questions": [{**good["questions"][0], "context_messages": [{"role": "system", "content": "x"}]}]})


def test_shared_classifier_rejects_non_object():
    with pytest.raises(ModelError):
        classify_route_timed("Question", FakeModel(None))
