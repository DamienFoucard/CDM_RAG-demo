"""Keep the displayed generation questions and evidence links tied to saved data."""
from html.parser import HTMLParser
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.links = []
        self.text = []
        self.groups = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("#"):
            self.links.append(attrs["href"][1:])
        if tag == "details" and "output-group" in attrs.get("class", "").split():
            self.groups.append(attrs)

    def handle_data(self, data):
        self.text.append(data)


def load_page():
    page = Page()
    page.feed((ROOT / "web/benchmark.html").read_text())
    return page


def test_questions_references_and_local_navigation():
    page = load_page()
    assert len(page.ids) == len(set(page.ids))
    assert set(page.links) <= set(page.ids)
    visible_text = " ".join(page.text)
    for dataset in ("answer-benchmark-v1.json", "paired-answer-challenge-candidate-v1.json"):
        for case in json.loads((ROOT / "evaluation" / dataset).read_text())["questions"]:
            assert case["question"] in visible_text
            assert case["reference_answer"] in visible_text


def test_each_generation_output_link_has_real_saved_test_sources():
    page = load_page()
    rows = json.loads((ROOT / "evaluation/results/generation-output-snapshot.json").read_text())["rows"]
    assert len(page.groups) == 10
    for group in page.groups:
        selected = [r for r in rows
                    if r["dataset_id"] == group["data-dataset"]
                    and r["model"] == group["data-model"]
                    and r["split"] == "test"
                    and (not group.get("data-arm") or r["arm"] == group["data-arm"])]
        assert len(selected) == (4 if group["data-dataset"] == "fixed-evidence" else 2)
        assert all(r["retrieved_sources"] for r in selected)
        assert all(r.get("retrieval_note") for r in selected)
