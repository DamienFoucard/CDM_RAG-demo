"""Check that the routing page and saved results refer to the actual test cases."""
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from app.decision_router import initial_query


ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.anchors = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"])
        if tag == "a" and values.get("href", "").startswith("#"):
            self.anchors.append(values["href"][1:])


def test_routing_page_navigation_and_script(tmp_path):
    html = (ROOT / "web/routing-benchmark.html").read_text()
    page = Page()
    page.feed(html)
    assert len(page.ids) == len(set(page.ids))
    assert set(page.anchors) <= set(page.ids)
    assert {"routing-results", "scope-results", "evidence-results", "routing-outputs"} <= set(page.ids)
    for other_page in ("benchmarks.html", "retrieval-benchmark.html", "benchmark.html"):
        assert 'href="/routing-benchmark"' in (ROOT / "web" / other_page).read_text()
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed to check the page script")
    script = re.search(r"<script>([\s\S]*?)</script>", html)
    assert script
    path = tmp_path / "routing-page.js"
    path.write_text(script.group(1))
    subprocess.run([node, "--check", str(path)], check=True, capture_output=True, text=True)


def test_home_model_settings_expose_separate_decision_and_query_models():
    html = (ROOT / "web/index.html").read_text()
    assert 'id="decision-model-select"' in html
    assert 'id="router-model-select"' in html
    assert "selected.decision_model||selected.router_model" in html
    assert "decision_model:$('decision-model-select').value" in html
    assert "router_model:$('router-model-select').value" in html


@pytest.mark.parametrize(
    ("dataset_file", "snapshot_file"),
    [("routing-decision-v1.json", "routing-benchmark-snapshot.json"),
     ("routing-scope-binary-v1.json", "routing-scope-binary-snapshot.json")],
)
def test_saved_question_only_runs_match_their_test_labels(dataset_file, snapshot_file):
    dataset = json.loads((ROOT / "evaluation" / dataset_file).read_text())
    snapshot = json.loads((ROOT / "evaluation/results" / snapshot_file).read_text())
    run = snapshot.get("run", snapshot)
    assert run["dataset_id"] == dataset["dataset_id"]
    assert run["split"] == "test"
    cases = {q["id"]: q for q in dataset["questions"] if q["split"] == "test"}
    assert cases and run["summaries"]
    for model, summary in run["summaries"].items():
        rows = [r for r in run["rows"] if r["model"] == model]
        assert len(rows) == summary["attempted"]
        assert len({r["case_id"] for r in rows}) == len(rows)
        for row in rows:
            case = cases[row["case_id"]]
            assert row["question"] == case["question"]
            if "expected_route" in case:
                assert row["expected_route"] == case["expected_route"]
            else:
                assert row["accepted_routes"] == case["accepted_routes"]


@pytest.mark.parametrize(
    ("dataset_file", "snapshot_file"),
    [("routing-decision-v1.json", "routing-evidence-aware-snapshot.json"),
     ("routing-scope-binary-v1.json", "routing-scope-evidence-aware-snapshot.json")],
)
def test_evidence_runs_keep_test_questions_and_bounded_search_outputs(dataset_file, snapshot_file):
    dataset = json.loads((ROOT / "evaluation" / dataset_file).read_text())
    snapshot = json.loads((ROOT / "evaluation/results" / snapshot_file).read_text())
    run = snapshot.get("run", snapshot)
    assert run["dataset_id"] == dataset["dataset_id"] and run["split"] == "test"
    assert run["search"]["reranker"] is None
    cases = {q["id"]: q for q in dataset["questions"] if q["split"] == "test"}
    for model, summary in run["summaries"].items():
        rows = [row for row in run["rows"] if row["model"] == model]
        assert len(rows) == summary["attempted"]
        assert len({row["case_id"] for row in rows}) == len(rows)
        for row in rows:
            case = cases[row["case_id"]]
            assert row["question"] == case["question"]
            assert row["initial_query"] == initial_query(case)
            assert len(row["initial_hits"]) <= run["search"]["top_k"]
            assert all(len(hit["text"]) <= run["search"]["snippet_chars"] for hit in row["initial_hits"])
            assert row["end_to_end_seconds"] >= max(row["search_seconds"], row["router_seconds"])
    recommendation = snapshot.get("recommendation")
    if recommendation:
        assert recommendation["model"] in run["summaries"]
