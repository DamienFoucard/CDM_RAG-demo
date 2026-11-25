"""The public drill-down must faithfully reflect the saved routing experiment."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from app.retrieval.decision_router import evidence_messages
from scripts.benchmark_router_decisions import accepted_routes, summarize
from scripts.export_evidence_router_snapshots import catalog_index, validate_and_sanitize


ROOT = Path(__file__).resolve().parents[1]
PAIRS = (
    ("routing-evidence-aware-snapshot.json", "routing-decision-v1.json"),
    ("routing-scope-evidence-aware-snapshot.json", "routing-scope-binary-v1.json"),
)


@pytest.mark.parametrize("snapshot_name,dataset_name", PAIRS)
def test_snapshot_preserves_sources_labels_and_exact_messages(snapshot_name, dataset_name):
    snapshot = json.loads((ROOT / "evaluation/results" / snapshot_name).read_text())
    raw = (ROOT / "evaluation" / dataset_name).read_bytes()
    dataset = json.loads(raw)
    cases = {item["id"]: item for item in dataset["questions"] if item["split"] == "test"}
    passages = catalog_index(snapshot)
    assert snapshot["dataset_sha256"] == hashlib.sha256(raw).hexdigest()
    assert snapshot["split"] == "test"
    assert set(snapshot["selected_case_ids"]) == set(cases)
    assert "host" not in snapshot
    assert "/Users/" not in json.dumps(snapshot, ensure_ascii=False)
    assert "All cases are searched before classification" in snapshot["caveat"]
    for row in snapshot["rows"]:
        case = cases[row["case_id"]]
        assert row["question"] == case["question"]
        assert row["context_messages"] == case.get("context_messages", [])
        assert row["accepted_routes"] == accepted_routes(case)
        assert len(row["initial_hits"]) <= 3
        for number, hit in enumerate(row["initial_hits"], 1):
            passage = passages[hit["id"]]
            assert hit["rank"] == number
            assert hit["text"] == passage["text"][:700]
            assert hit["title"] == passage.get("title", "")
            assert hit["uri"] == passage.get("uri", "")
        if row["status"] == "scored":
            messages = evidence_messages(case, row["initial_hits"])
            assert row["messages_sha256"] == hashlib.sha256(
                json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            assert row["correct"] == (row["predicted_route"] in row["accepted_routes"])
    for tag, recorded in snapshot["summaries"].items():
        computed = summarize([r for r in snapshot["rows"] if r["model"] == tag])
        assert all(recorded[k] == value for k, value in computed.items())


def test_provisional_role_recommendation_is_not_duplicated():
    main = json.loads((ROOT / "evaluation/results" / PAIRS[0][0]).read_text())
    scope = json.loads((ROOT / "evaluation/results" / PAIRS[1][0]).read_text())
    assert main["recommendation"]["model"] == "qwen3.5:9b-q4_K_M"
    assert "27B model was not tested" in main["recommendation"]["reason"]
    assert "recommendation" not in scope


def test_exporter_rejects_a_tampered_retrieval_excerpt():
    run_path = ROOT / "data/macstudio-runs/routing-evidence-aware-test.json"
    if not run_path.exists():
        pytest.skip("Local benchmark artifacts are not part of the repository")
    baseline_path = ROOT / "data/macstudio-runs/routing-decisions-test-small.json"
    dataset_path = ROOT / "evaluation/routing-decision-v1.json"
    raw, baseline_raw, dataset_raw = run_path.read_bytes(), baseline_path.read_bytes(), dataset_path.read_bytes()
    run, baseline, dataset = json.loads(raw), json.loads(baseline_raw), json.loads(dataset_raw)
    tampered = copy.deepcopy(run)
    tampered["rows"][0]["initial_hits"][0]["text"] += " invented"
    with pytest.raises(ValueError, match="differs from pinned catalog"):
        validate_and_sanitize(tampered, raw, dataset, dataset_raw, baseline, baseline_raw, catalog_index(run))

