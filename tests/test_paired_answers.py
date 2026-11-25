from types import SimpleNamespace
import json
from pathlib import Path

from scripts import benchmark_paired_answers as paired
from app.corpus import render_chunks


def test_two_kinds_policy_keeps_complementary_evidence_in_rank_order():
    def passage(item_id, entity, kind):
        return {"id": item_id, "entity_id": entity, "source_id": entity, "source_type": "entity",
                "kind": kind, "text": item_id, "uri": "https://example.com"}

    candidates = [passage("a-entity", "A", "entity"),
                  passage("a-attributes", "A", "attributes"),
                  passage("a-other-attributes", "A", "attributes"),
                  passage("a-relationship", "A", "relationship"),
                  passage("b-relationship", "B", "relationship"),
                  passage("b-entity", "B", "entity")]
    default, _ = paired.select_context(candidates, 6)
    two_kinds, _ = paired.select_context(candidates, 6, "two_kinds_per_source")
    assert [item["passage_id"] for item in default] == ["a-entity", "b-relationship"]
    assert [item["passage_id"] for item in two_kinds] == [
        "a-entity", "a-attributes", "b-relationship", "b-entity"]
    bounded, _ = paired.select_context(candidates, 3, "two_kinds_per_source")
    assert [item["passage_id"] for item in bounded] == ["a-entity", "a-attributes", "b-relationship"]


def test_context_policy_rejects_unknown_strategy():
    import pytest
    with pytest.raises(ValueError, match="Unknown context"):
        paired.select_context([], 6, "all")


def test_selected_context_policy_is_applied_to_both_arms(monkeypatch):
    policies = []
    models = SimpleNamespace(router_model="router", answer_model="answer", embedding_model="embed",
                             unload_if_loaded=lambda name: None, runtime_info=lambda name: None)
    class Index:
        def search(self, query, *, limit, mode):
            return [{"id": query, "entity_id": query, "source_id": query, "source_type": "entity",
                     "kind": "entity", "text": query, "uri": "https://example.com"}]
    original = paired.select_context
    def select(candidates, limit, strategy):
        policies.append(strategy)
        return original(candidates, limit, strategy)
    monkeypatch.setattr(paired, "select_context", select)
    monkeypatch.setattr(paired, "rewrite_search_query", lambda *args: "rewritten")
    monkeypatch.setattr(paired, "cited_answer", lambda q, s, c, m: ("answer [R1]", m.answer_model))
    monkeypatch.setattr(paired, "_judge_answer", lambda *args: {"score": 4, "rationale": "ok"})
    case = {"id": "x", "question": "raw", "reference_answer": "reference", "expected_entity_ids": ["raw"]}
    result = paired.run_case(case, Index(), models, "judge", "two_kinds_per_source")
    assert policies == ["two_kinds_per_source", "two_kinds_per_source"]
    assert all(item["status"] == "scored" for item in result["arms"].values())


def test_paired_answers_use_distinct_retrieval_with_same_answer_and_judge(monkeypatch):
    calls = []
    models = SimpleNamespace(router_model="router", answer_model="answer", embedding_model="embed",
                             unload_if_loaded=lambda name: None, runtime_info=lambda name: {"model": name})
    class Index:
        def search(self, query, *, limit, mode):
            calls.append(("search", query, limit, mode))
            return [{"id": query, "entity_id": query, "source_id": query, "source_type": "entity",
                     "kind": "entity", "text": query, "uri": "https://example.com"}]
    monkeypatch.setattr(paired, "rewrite_search_query", lambda *args: "rewritten")
    monkeypatch.setattr(paired, "cited_answer", lambda q, s, c, m: ("answer [R1]", m.answer_model))
    monkeypatch.setattr(paired, "_judge_answer", lambda q, ref, ans, exp, cited, s, m, judge:
                        calls.append(("judge", q, ref, judge, s[0]["text"])) or {"score": 4, "rationale": "ok"})
    case = {"id": "x", "question": "raw question", "reference_answer": "reference", "expected_entity_ids": ["raw question"]}
    result = paired.run_case(case, Index(), models, "judge")
    assert calls == [("search", "raw question", 30, "hybrid"),
                     ("judge", "raw question", "reference", "judge", "raw question"),
                     ("search", "rewritten", 30, "hybrid"),
                     ("judge", "raw question", "reference", "judge", "rewritten")]
    assert result["arms"]["raw"]["generator"] == result["arms"]["rewrite"]["generator"] == "answer"
    assert paired.paired_summary([result])["tied"] == 1


def test_rewrite_failure_is_not_counted_as_a_paired_result(monkeypatch):
    models = SimpleNamespace(router_model="router", answer_model="answer", embedding_model="embed",
                             unload_if_loaded=lambda name: None, runtime_info=lambda name: None)
    class Index:
        def search(self, query, *, limit, mode):
            return []
    def fail(*args):
        raise RuntimeError("router failed")
    monkeypatch.setattr(paired, "rewrite_search_query", fail)
    case = {"id": "x", "question": "q", "reference_answer": "r", "expected_entity_ids": ["a"]}
    result = paired.run_case(case, Index(), models, "judge")
    assert result["arms"]["rewrite"]["stage"] == "rewrite"
    assert "router failed" in result["arms"]["rewrite"]["error"]
    assert result["arms"]["raw"]["stage"] == "retrieval"
    assert paired.paired_summary([result])["paired_scored"] == 0


def test_challenge_references_exist_in_pinned_corpus():
    root = Path(__file__).resolve().parents[1]
    dataset = json.loads((root / "evaluation/paired-answer-challenge-candidate-v1.json").read_text())
    catalog = json.loads((root / "seed/catalog.json").read_text())
    passages = {item["id"]: item for item in render_chunks(catalog)}
    assert dataset["source_commit"] == catalog["source_commit"]
    assert len(dataset["questions"]) == len({q["id"] for q in dataset["questions"]}) == 6
    for case in dataset["questions"]:
        supporting = [passages[item] for item in case["evidence_passage_ids"]]
        assert set(case["expected_entity_ids"]) <= {item.get("entity_id") for item in supporting}
        assert case["reference_answer"] and case["split"] in {"development", "test"}
    assert "104800005" in passages["RetailBankingCoreDataModel::CustomerFinancialHolding:trait-applications:attributes:2"]["text"]


def test_repository_metadata_handles_container_without_git(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git not installed")
    monkeypatch.setattr(paired.subprocess, "run", missing)
    assert paired.repository_metadata(None) == {
        "repository_commit": None, "repository_commit_source": "unavailable",
        "git_status_porcelain": None, "git_status": "unknown"}
    supplied = paired.repository_metadata("a" * 40)
    assert supplied["repository_commit"] == "a" * 40
    assert supplied["repository_commit_source"] == "operator_supplied"
    assert supplied["git_status"] == "unknown" and supplied["git_status_porcelain"] is None


def test_repository_metadata_rejects_invalid_supplied_revision():
    import pytest
    with pytest.raises(ValueError, match="40-character"):
        paired.repository_metadata("not-a-commit")
