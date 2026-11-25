import subprocess

from scripts import review_saved_claims as replay


def test_replays_saved_passages_without_sending_human_annotations():
    received = []
    original_sources = [{"citation": "R1", "text": "saved evidence"}]
    original_contexts = [{"citation": "R1", "text": "saved context"}]
    arm = {"status": "scored", "raw_answer": "saved draft [R1]",
           "sources": original_sources, "contexts": original_contexts}
    payload = {"rows": [{"id": "paired-dev-card-customer", "question": "saved question",
                          "reference_answer": "hidden reference", "arms": {"raw": arm}}]}
    def reviewer(question, answer, sources, contexts, models):
        received.append((question, answer, sources, contexts, models))
        return {"decision": "accept"}

    observed = replay.replay([payload], reviewer, "judge")
    assert received == [("saved question", "saved draft [R1]", original_sources, original_contexts, "judge")]
    assert observed[0]["accepted_despite_observation"] is True
    assert observed[0]["human_observation"] not in repr(received)
    assert "hidden reference" not in repr(received)


def test_review_errors_are_counted_as_errors_not_rejections():
    def reviewer(*args):
        raise RuntimeError("model returned invalid JSON")
    payload = {"rows": [{"id": "paired-dev-card-customer", "question": "q", "arms": {
        "raw": {"status": "scored", "raw_answer": "draft", "sources": [], "contexts": []}}}]}
    result = replay.replay([payload], reviewer, "judge")
    assert result[0]["status"] == "error"
    assert "model returned invalid JSON" in result[0]["error"]
    assert "accepted_despite_observation" not in result[0]


def test_can_load_committed_baseline_and_current_logic(tmp_path, monkeypatch):
    # A clean distribution need not contain the original repository's history.
    # Create a real isolated revision so the git-backed loader is still tested.
    source = (replay.ROOT / "app/rag.py").read_bytes()
    (tmp_path / "app").mkdir()
    (tmp_path / "app/rag.py").write_bytes(source)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "app/rag.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.test",
                    "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    baseline, baseline_hash = replay.load_review_function("HEAD")
    current, current_hash = replay.load_review_function(None)
    assert callable(baseline) and callable(current)
    assert baseline_hash == current_hash and len(baseline_hash) == 64
