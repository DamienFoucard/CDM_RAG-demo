"""The drill-down snapshots should identify their source and not invent rows."""
import json
from pathlib import Path

from app.corpus import load_catalog, render_chunks
from app.graph import SchemaGraph
from scripts.export_benchmark_snapshots import clean_source

ROOT = Path(__file__).resolve().parents[1]


def test_export_drops_unrelated_source_metadata():
    source = {"id": "example", "text": "saved passage", "private_path": "/Users/example", "secret": "x"}
    assert clean_source(source) == {"id": "example", "text": "saved passage"}


def test_generation_questions_keep_the_original_split_and_reference():
    snapshot = json.loads((ROOT / "evaluation/results/generation-output-snapshot.json").read_text())
    original = json.loads((ROOT / "evaluation/answer-benchmark-v1.json").read_text())
    paired = json.loads((ROOT / "evaluation/paired-answer-challenge-candidate-v1.json").read_text())
    datasets = {
        "answer-benchmark-v1": {q["id"]: q for q in original["questions"]},
        "paired-rag-revised": {q["id"]: q for q in paired["questions"]},
        "fixed-evidence": {q["id"]: q for q in paired["questions"]},
    }
    assert snapshot["rows"]
    seen = set()
    for row in snapshot["rows"]:
        key = (row["dataset_id"], row["model"], row["case_id"], row["arm"])
        assert key not in seen
        seen.add(key)
        original_question = datasets[row["dataset_id"]][row["case_id"]]
        assert row["split"] == original_question["split"]
        assert row["question"] == original_question["question"]
        assert row["reference_answer"] == original_question["reference_answer"]
        if row["status"] in {"not_run", "error"}:
            assert not row["answer"]


def test_retrieval_snapshot_preserves_saved_passage_identity_and_text():
    snapshot = json.loads((ROOT / "evaluation/results/retrieval-qwen35-27b-output-snapshot.json").read_text())
    dataset = json.loads((ROOT / "evaluation/reformulation-challenge-candidate-v1.json").read_text())
    cases = {q["id"]: q for q in dataset["questions"]}
    catalog = load_catalog(ROOT / "seed/catalog.json")
    SchemaGraph(catalog["entities"], catalog["source_commit"], semantics=catalog.get("semantics", {}))
    by_id = {p["id"]: p for p in render_chunks(catalog)}
    assert snapshot["catalog_revision"] == dataset["source_commit"] == catalog["source_commit"]
    assert len(snapshot["rows"]) == 18
    for row in snapshot["rows"]:
        case = cases[row["question_id"]]
        assert row["split"] == case["split"] == "test"
        assert row["question"] == case["question"]
        assert row["target_ids"] == case["positive_passage_ids"]
        assert row["ranked_passage_ids"] == [p["id"] for p in row["ranked_passages"]]
        for passage in row["ranked_passages"]:
            assert passage["text"] == by_id[passage["id"]]["text"]
            assert passage["uri"] == by_id[passage["id"]]["uri"]
