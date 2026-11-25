import copy
import os
from types import SimpleNamespace

import pytest

from scripts import compare_fixed_evidence_answers as fixed


def saved_pair():
    return {"rows": [{"id": "case-1", "question": "What is it?", "reference_answer": "A fact.",
                      "arms": {
                          "raw": {"status": "scored", "query": "raw", "contexts": [{"text": "raw evidence"}],
                                  "sources": [{"citation": "R1", "text": "raw evidence"}]},
                          "rewrite": {"status": "scored", "query": "rewrite",
                                      "contexts": [{"text": "rewrite evidence"}],
                                      "sources": [{"citation": "R1", "text": "rewrite evidence"}]},
                      }}]}


def test_generators_receive_same_exact_saved_evidence_independently(monkeypatch):
    payload = saved_pair()
    before = copy.deepcopy(payload)
    cases = fixed.collect_cases([payload])
    seen = []

    def answer(question, sources, contexts, models):
        seen.append((question, copy.deepcopy(sources), copy.deepcopy(contexts), models.answer_model))
        # A model or helper must not be able to alter the stored evidence for the next model.
        contexts[0]["text"] = "mutated"
        return "A fact [R1].", models.answer_model

    monkeypatch.setattr(fixed, "cited_answer", answer)
    monkeypatch.setattr(fixed, "format_entity_citations", lambda raw, sources: (raw, sources))
    for model in ("9b", "12b"):
        result = fixed.generate_case(cases[0], SimpleNamespace(answer_model=model, runtime_info=lambda _: None))
        assert result["status"] == "generated"
        assert result["evidence_sha256"] == cases[0]["evidence_sha256"]
    assert seen[0][:3] == seen[1][:3]
    assert seen[0][2][0]["text"] == "raw evidence"
    assert payload == before


def test_incomplete_or_duplicated_saved_evidence_is_rejected():
    payload = saved_pair()
    payload["rows"][0]["arms"]["raw"]["status"] = "error"
    with pytest.raises(ValueError, match="not scored"):
        fixed.collect_cases([payload])
    with pytest.raises(ValueError, match="unique"):
        fixed.collect_cases([saved_pair(), saved_pair()])


def test_case_selection_keeps_both_arms_in_saved_order_without_mutation():
    first = saved_pair()
    second = saved_pair()
    second["rows"][0]["id"] = "case-2"
    before = copy.deepcopy([first, second])
    cases = fixed.collect_cases([first, second])
    selected = fixed.select_cases(cases, ["case-2"])
    assert [(case["case_id"], case["arm"]) for case in selected] == [
        ("case-2", "raw"), ("case-2", "rewrite")]
    assert [first, second] == before
    assert fixed.select_cases(cases, None) == cases
    assert [(case["case_id"], case["arm"]) for case in fixed.select_cases(cases, ["case-2", "case-1"])] == [
        ("case-1", "raw"), ("case-1", "rewrite"), ("case-2", "raw"), ("case-2", "rewrite")]


def test_arm_selection_keeps_only_requested_arm_and_saved_evidence():
    cases = fixed.collect_cases([saved_pair()])
    before = copy.deepcopy(cases)
    selected = fixed.select_cases(cases, ["case-1"], ["rewrite"])
    assert len(selected) == 1
    assert selected[0]["arm"] == "rewrite"
    assert selected[0]["evidence_sha256"] == cases[1]["evidence_sha256"]
    assert cases == before
    assert fixed.select_cases(cases, None, ["rewrite", "raw"]) == cases


@pytest.mark.parametrize("arms", [[], ["raw", "raw"], ["missing"]])
def test_arm_selection_rejects_empty_duplicate_and_unknown_values(arms):
    with pytest.raises(ValueError):
        fixed.select_cases(fixed.collect_cases([saved_pair()]), None, arms)


@pytest.mark.parametrize("selection", [[], ["case-1", "case-1"], ["missing"]])
def test_case_selection_rejects_empty_duplicate_and_unknown_ids(selection):
    with pytest.raises(ValueError):
        fixed.select_cases(fixed.collect_cases([saved_pair()]), selection)


def test_failed_generation_and_extractive_fallback_are_not_counted_as_answers(monkeypatch):
    case = fixed.collect_cases([saved_pair()])[0]
    model = SimpleNamespace(answer_model="9b", runtime_info=lambda _: None)
    monkeypatch.setattr(fixed, "cited_answer", lambda *args: ("A fact [R1]", "extractive-fallback"))
    monkeypatch.setattr(fixed, "format_entity_citations", lambda raw, sources: (raw, sources))
    assert fixed.generate_case(case, model)["status"] == "fallback"
    def fail(*args):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(fixed, "cited_answer", fail)
    assert fixed.generate_case(case, model)["status"] == "error"


def test_failed_generation_preserves_current_native_counters_but_not_stale_ones(monkeypatch):
    case = fixed.collect_cases([saved_pair()])[0]
    local = SimpleNamespace(timings={"eval_count": 999})
    model = SimpleNamespace(answer_model="9b", _chat_local=local)

    def fail_after_response(*args):
        assert local.timings is None
        local.timings = {"eval_count": 4096, "prompt_eval_count": 1024}
        raise RuntimeError("empty model response")

    monkeypatch.setattr(fixed, "cited_answer", fail_after_response)
    result = fixed.generate_case(case, model)
    assert result["status"] == "error"
    assert result["error_type"] == "RuntimeError"
    assert result["native_timing"] == {"eval_count": 4096, "prompt_eval_count": 1024}
    local.timings["eval_count"] = 1
    assert result["native_timing"]["eval_count"] == 4096

    def fail_before_response(*args):
        assert local.timings is None
        raise ValueError("before model call")

    monkeypatch.setattr(fixed, "cited_answer", fail_before_response)
    result = fixed.generate_case(case, model)
    assert result["error_type"] == "ValueError"
    assert result["native_timing"] is None


def test_checkpoint_is_private(tmp_path):
    path = tmp_path / "result.json"
    fixed.write_result(path, {"status": "running"})
    assert os.stat(path).st_mode & 0o777 == 0o600
