import pytest

from scripts.judge_retrieval import summarize, validate_judgments
from scripts.compare_retrieval import paired_comparison
from scripts.audit_judgments import audit


def test_judge_must_grade_every_candidate_once():
    valid = {"judgments": [{"candidate_id": "C01", "grade": 2, "reason": "Direct evidence"},
                           {"candidate_id": "C02", "grade": 0, "reason": "Unrelated"}]}
    assert validate_judgments(valid, {"C01", "C02"})["C01"]["grade"] == 2
    with pytest.raises(ValueError, match="omitted"):
        validate_judgments({"judgments": valid["judgments"][:1]}, {"C01", "C02"})
    with pytest.raises(ValueError, match="duplicate"):
        validate_judgments({"judgments": valid["judgments"] + valid["judgments"][:1]}, {"C01", "C02"})
    with pytest.raises(ValueError, match="invalid"):
        validate_judgments({"judgments": [{"candidate_id": "C01", "grade": True, "reason": "yes"}]}, {"C01"})


def test_judge_metrics_use_pooled_ideal_and_penalize_empty_slots():
    rows = [{"judgments": {"A": {"grade": 2}, "B": {"grade": 1}},
             "rankings": {"first": ["A", "B"], "second": ["B", "A"]}}]
    results = summarize(rows, ["first", "second"])
    assert results["first"]["ndcg_at_6"] == 1
    assert results["second"]["ndcg_at_6"] < 1
    assert results["first"]["precision_at_6"] == pytest.approx(2 / 6, abs=0.0001)


def test_paired_comparison_requires_identical_questions_and_labels():
    case = {"id": "q", "question": "Find A", "positive_entity_ids": ["M::A"],
            "variants": {"hybrid": {"mrr_at_6": 0.5}}}
    baseline = {"dataset_sha256": "abc", "source_commit": "src", "split": "test",
                "embedding_model": "old", "cases": [case]}
    candidate = {**baseline, "embedding_model": "new", "cases": [
        {**case, "variants": {"hybrid": {"mrr_at_6": 1.0}}}]}
    result = paired_comparison(baseline, candidate, "hybrid", samples=50)
    assert (result["wins"], result["mrr_delta"]) == (1, 0.5)
    candidate["cases"][0]["positive_entity_ids"] = ["M::B"]
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(baseline, candidate, "hybrid", samples=50)


def test_judge_audit_reports_disagreement_instead_of_relabeling():
    result = {"dataset_sha256": "abc", "split": "test", "judge_model": "local", "judge_model_digest": "v1",
              "cases": [{"id": "q", "question": "Find A", "positive_entity_ids": ["M::A"],
                         "judgments": {"M::A": {"grade": 0, "reason": "mistaken"}}}]}
    report = audit(result)
    assert report["judge_direct_count"] == 0
    assert report["anchor_disagreements"][0]["entity_id"] == "M::A"
