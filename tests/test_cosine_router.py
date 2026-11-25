"""The cosine-only baseline must calibrate on development, not guess routes."""
import hashlib
import json
import math
from pathlib import Path

import pytest

from app.index import SchemaIndex
from app.ollama import ModelError
from scripts.benchmark_cosine_router import (choose_threshold, cosine_probe_with_passage, probe_case,
                                              score_test, scope_development_cases)
from scripts.export_cosine_router_snapshot import catalog_fragments

ROOT = Path(__file__).resolve().parents[1]


def sample(similarity, expected_search, status="scored"):
    return {"similarity": similarity, "expected_search": expected_search, "status": status}


def test_threshold_optimizes_balanced_accuracy_and_breaks_ties_conservatively():
    calibration = choose_threshold([sample(0.7, True), sample(0.6, True),
                                    sample(0.6, False), sample(0.2, False)])
    # 0.6 and 0.7 both have balanced accuracy .75; the higher cutoff is safer.
    assert calibration["threshold"] == 0.7
    assert calibration["balanced_accuracy"] == pytest.approx(0.75)
    assert calibration["search_recall"] == pytest.approx(0.5)
    assert calibration["no_search_recall"] == 1


def test_calibration_can_choose_no_search_for_a_complete_tie():
    calibration = choose_threshold([sample(0.5, True), sample(0.5, False)])
    assert calibration["threshold"] > 0.5
    assert calibration["balanced_accuracy"] == 0.5


@pytest.mark.parametrize("rows", [[], [sample(0.5, True)],
                                [sample(0.5, True), sample(0.2, False, "error")],
                                [sample(math.nan, True), sample(0.2, False)]])
def test_invalid_or_incomplete_development_cannot_calibrate(rows):
    with pytest.raises(ValueError):
        choose_threshold(rows)


def test_binary_gate_does_not_claim_it_can_generate_rewrites():
    rows = [
        {"status": "scored", "similarity": 0.8, "accepted_routes": ["raw_search"], "elapsed_seconds": 0.1},
        {"status": "scored", "similarity": 0.7, "accepted_routes": ["rewrite_search"], "elapsed_seconds": 0.1},
        {"status": "scored", "similarity": 0.2, "accepted_routes": ["no_search"], "elapsed_seconds": 0.1},
        {"status": "error", "similarity": None, "accepted_routes": ["no_search"], "elapsed_seconds": 0.1},
    ]
    result = score_test(rows, 0.5)
    assert [row["predicted_route"] for row in rows] == ["raw_search", "raw_search", "no_search", None]
    assert result["correct"] == 2  # rewrite example is not a correct three-way decision
    assert result["search_gate_correct"] == 3  # it is a correct binary search decision
    assert result["errors"] == 1


def test_probe_embeds_only_the_raw_conversation_and_question():
    class Embedder:
        def embed(self, questions, *, query):
            assert query is True
            self.questions = questions
            return [[0.1, 0.2]]

    class Collection:
        def query(self, *, query_embeddings, n_results, include):
            assert query_embeddings == [[0.1, 0.2]] and n_results == 1 and include == ["distances"]
            return {"ids": [["passage::fragment:1"]], "distances": [[0.25]]}

    class Index:
        models = Embedder()
        collection = Collection()
        vector_to_chunk = {"passage::fragment:1": "passage"}
        by_id = {"passage": {"id": "passage", "title": "Test passage", "uri": "https://example.test"}}
        embedding_records = [{"id": "passage::fragment:1", "text": "Exact embedded fragment"}]

    class Model:
        def runtime_info(self, _):
            return {"runner": "test"}

    case = {"id": "x", "split": "development", "question": "Which field?",
            "context_messages": [{"role": "user", "content": "The Account schema."}],
            "expected_route": "rewrite_search", "rationale": "SECRET GOLD RATIONALE"}
    index = Index()
    row = probe_case(case, "dataset", index, Model())
    assert index.models.questions == ["User: The Account schema.\nUser: Which field?"]
    assert "SECRET GOLD" not in index.models.questions[0] and "rewrite_search" not in index.models.questions[0]
    assert row["similarity"] == 0.75 and row["expected_search"] is True
    assert row["nearest_fragment_id"] == "passage::fragment:1"
    assert row["nearest_passage_id"] == "passage"
    assert row["nearest_fragment_text"] == "Exact embedded fragment"
    # Compare with the application's actual semantic-probe formula on the same vector/distance.
    assert row["similarity"] == SchemaIndex.semantic_probe(index, index.models.questions[0])


def test_empty_nearest_passage_is_a_model_error():
    class Embedder:
        def embed(self, questions, *, query):
            return [[1.0]]

    class Collection:
        def query(self, **kwargs):
            return {"ids": [[]], "distances": [[]]}

    class Index:
        models = Embedder()
        collection = Collection()

    with pytest.raises(ModelError):
        cosine_probe_with_passage(Index(), "question")


def test_scope_development_comes_only_from_the_checked_source():
    source = {"dataset_id": "scope", "cases": [
        {"id": "d", "split": "development", "question": "Schema?", "allow_schema_search": True},
        {"id": "t", "split": "test", "question": "Balance?", "allow_schema_search": False},
    ]}
    raw = json.dumps(source).encode()
    scope = {"dataset_id": "derived", "source_dataset": "scope", "source_sha256": hashlib.sha256(raw).hexdigest(),
             "questions": [{"id": "t", "source_id": "t", "split": "test", "question": "Balance?",
                            "accepted_routes": ["no_search"]}]}
    development = scope_development_cases(scope, source, raw)
    assert [case["id"] for case in development] == ["d"]
    assert development[0]["accepted_routes"] == ["raw_search", "rewrite_search"]
    scope["source_sha256"] = "wrong"
    with pytest.raises(ValueError):
        scope_development_cases(scope, source, raw)


def test_exported_test_decisions_and_fragments_match_the_pinned_index():
    snapshot = json.loads((ROOT / "evaluation/results/routing-cosine-snapshot.json").read_text())
    fragments = catalog_fragments(snapshot)
    threshold = snapshot["calibration"]["threshold"]
    for name in ("routing-decision-v1.json", "routing-scope-binary-v1.json"):
        dataset = json.loads((ROOT / "evaluation" / name).read_text())
        expected = {q["id"]: q for q in dataset["questions"] if q["split"] == "test"}
        group = snapshot["datasets"][dataset["dataset_id"]]
        assert {row["case_id"] for row in group["rows"]} == set(expected)
        for row in group["rows"]:
            case = expected[row["case_id"]]
            assert row["question"] == case["question"]
            assert row["accepted_routes"] == case.get("accepted_routes", [case.get("expected_route")])
            assert fragments[row["nearest_fragment_id"]] == (row["nearest_passage_id"], row["nearest_fragment_text"])
            assert row["predicted_route"] == ("raw_search" if row["similarity"] >= threshold else "no_search")
