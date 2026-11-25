"""Integrity checks for the source-derived routing-only benchmark."""
import json
import hashlib
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_routing_decisions_are_balanced_and_schema_sources_exist():
    data = json.loads((ROOT / "evaluation/routing-decision-v1.json").read_text())
    catalog = json.loads((ROOT / "seed/catalog.json").read_text())
    entity_ids = {entity["id"] for entity in catalog["entities"]}
    routes = {"raw_search", "rewrite_search", "no_search"}
    cases = data["questions"]

    assert data["source_commit"] == catalog["source_commit"]
    assert len({case["id"] for case in cases}) == len(cases)
    assert Counter((case["split"], case["expected_route"]) for case in cases) == {
        (split, route): count
        for split, count in (("development", 3), ("test", 6))
        for route in routes
    }
    for case in cases:
        assert case["question"].strip()
        assert set(case["source_ids"]) <= entity_ids
        assert bool(case["source_ids"]) == (case["expected_route"] != "no_search")
        if case["expected_route"] == "rewrite_search":
            assert case["context_messages"]
        assert all(message["role"] in {"user", "assistant"} and message["content"].strip()
                   for message in case["context_messages"])


def test_scope_routing_is_an_exact_binary_derivation():
    source_path = ROOT / "evaluation/scope-questions-v1.json"
    source = json.loads(source_path.read_text())
    result = json.loads((ROOT / "evaluation/routing-scope-binary-v1.json").read_text())
    original = [item for item in source["cases"] if item["split"] == "test"]

    assert result["source_dataset"] == source["dataset_id"]
    assert result["source_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert len(result["questions"]) == len(original) == 32
    for row, source_row in zip(result["questions"], original, strict=True):
        assert row["id"] == row["source_id"] == source_row["id"]
        assert row["question"] == source_row["question"]
        assert row["source_expected"] == source_row["expected"]
        assert row["allow_schema_search"] is source_row["allow_schema_search"]
        assert "expected_route" not in row  # The source has no raw-vs-rewrite label.
        assert row["accepted_routes"] == (
            ["raw_search", "rewrite_search"] if source_row["allow_schema_search"] else ["no_search"]
        )

