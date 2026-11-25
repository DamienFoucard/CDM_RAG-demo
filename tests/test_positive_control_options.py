import hashlib
import json
import sys
from types import SimpleNamespace

from scripts import replay_reviewer_positive_controls as positive


def test_positive_control_cli_passes_opt_in_settings_and_records_provenance(tmp_path, monkeypatch):
    source = tmp_path / "saved.sse"
    source.write_bytes(b"saved evidence")
    output = tmp_path / "results.json"
    monkeypatch.setattr(sys, "argv", ["replay", "--input", str(source), "--output", str(output),
                                     "--judge", "gpt-oss:20b", "--reasoning-effort", "low",
                                     "--max-output-tokens", "2048"])
    controls = [{"question": "q", "answer": "answer [R1]", "sources": [{"text": "evidence"}],
                 "contexts": [], "draft_model": "draft", "question_id": "id"}]
    monkeypatch.setattr(positive, "collect_controls", lambda _: controls)
    def review(question, answer, sources, contexts, models):
        assert (question, answer, sources, contexts) == ("q", "answer [R1]", [{"text": "evidence"}], [])
        payload = models._chat_payload(models.judge_model, [{"role": "user", "content": question}],
                                       stream=True, as_json=True, max_tokens=600, schema={"type": "object"})
        assert payload["think"] == "low" and payload["options"]["num_predict"] == 2048
        return {"decision": "accept"}
    monkeypatch.setattr(positive, "load_review_function", lambda _: (review, "review-hash"))
    monkeypatch.setattr(positive.httpx, "get", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"models": [{"name": "gpt-oss:20b", "digest": "digest"}]}))
    monkeypatch.setattr(positive.ComparisonOllama, "unload_if_loaded", lambda *args: None)
    positive.main()
    result = json.loads(output.read_text())
    assert result["model_settings"]["think"] == "low"
    assert result["model_settings"]["max_output_tokens"] == 2048
    assert result["review_source_sha256"] == "review-hash"
    assert result["input_sha256"] == hashlib.sha256(b"saved evidence").hexdigest()
    assert "scripts/comparison_ollama.py" in result["implementation_sha256"]
    assert result["judges"][0]["rows"][0]["status"] == "reviewed"
