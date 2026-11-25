import hashlib
import json

import pytest

from scripts.export_blind_answer_review import make_review
from scripts.replay_reviewer_positive_controls import CONTROLS, collect_controls, replay


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def test_blind_export_hashes_and_blinds_the_model():
    evidence = {"contexts": [{"entity_id": "X"}], "sources": [{"citation": "R1", "text": "Evidence ``` example"}]}
    digest = content_hash(evidence)
    case = {"case_id": "case", "arm": "rewrite", "question": "Question?", "reference_answer": "Expected",
            "evidence_sha256": digest, **evidence}
    rows = [{"case_id": "case", "arm": "rewrite", "model": model, "evidence_sha256": digest,
             "status": "generated", "raw_answer": answer} for model, answer in (("model_alpha", "Yes [R1]."),
                                                                                 ("model_beta", "No [R1]."))]
    payload = {"cases": [case], "rows": rows}
    review, key = make_review(payload, 42)
    assert "model_alpha" not in review and "model_beta" not in review and "rewrite" not in review
    assert "Yes [R1]." in review and "No [R1]." in review and "Evidence ``` example" in review
    assert "````\nEvidence ``` example\n````" in review
    assert {item["model"] for item in key["items"]} == {"model_alpha", "model_beta"}
    assert make_review(payload, 42) == (review, key)
    case["sources"][0]["text"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        make_review(payload, 42)


def test_blind_export_marks_failed_generations():
    evidence = {"contexts": [], "sources": [{"citation": "R1", "text": "A"}]}
    digest = content_hash(evidence)
    case = {"case_id": "1", "arm": "raw", "evidence_sha256": digest, **evidence}
    row = {"case_id": "1", "arm": "raw", "model": "x", "status": "error", "raw_answer": "not valid",
           "evidence_sha256": digest}
    review, _ = make_review({"cases": [case], "rows": [row]}, 0)
    assert "Answer unavailable" in review and "not valid" not in review


def test_positive_control_collection_and_replay_preserve_errors_without_expected_labels():
    rows = [{"model": model, "question_id": question, "question": question, "raw_answer": "Supported [R1]",
             "retrieved_sources": [{"citation": "R1", "text": "Supported"}], "error": None}
            for model, question in CONTROLS]
    raw = ("data: " + json.dumps({"result": {"rows": rows}}) + "\n").encode()
    controls = collect_controls(raw)
    seen = []
    def reviewer(question, answer, sources, contexts, models):
        seen.append((question, answer, sources, contexts, models))
        if question == "bank-03":
            raise ValueError("invalid output")
        return {"decision": "accept"}
    results = replay(controls, reviewer, "judge")
    assert len(seen) == 4 and all(item[3] == [] and item[4] == "judge" for item in seen)
    assert all("expected" not in str(item) for item in seen)
    assert [row["status"] for row in results] == ["reviewed", "reviewed", "error", "reviewed"]
    assert results[2]["error"] == "ValueError: invalid output"
    assert len({row["control_sha256"] for row in results}) == 3  # shared bank-01 draft is identical in fixture
    with pytest.raises(ValueError, match="Missing controls"):
        collect_controls(("data: " + json.dumps({"result": {"rows": rows[:-1]}}) + "\n").encode())
