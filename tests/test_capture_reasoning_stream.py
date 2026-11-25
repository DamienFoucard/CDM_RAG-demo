import json
import sys

import httpx
import pytest

from scripts import capture_reasoning_stream as capture


@pytest.mark.parametrize("timeout", [False, True])
def test_capture_preserves_reasoning_and_distinguishes_idle_timeout(tmp_path, monkeypatch, timeout):
    source = tmp_path / "run.json"
    source.write_text(json.dumps({"cases": [{"case_id": "case", "arm": "rewrite"}]}))
    output = tmp_path / "stream.ndjson"
    monkeypatch.setattr(capture, "answer_messages", lambda case, model: [{"role": "user", "content": "Question"}])
    monkeypatch.setattr(sys, "argv", ["capture_reasoning_stream.py", "--input", str(source),
                                     "--case-id", "case", "--arm", "rewrite", "--model", "local",
                                     "--output", str(output)])

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield json.dumps({"message": {"thinking": "checking evidence"}, "done": False})
            if timeout:
                raise httpx.ReadTimeout("idle")
            yield json.dumps({"message": {"content": "answer"}, "done": False})
            yield json.dumps({"message": {}, "done": True, "done_reason": "stop"})

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def stream(self, method, url, json):
            assert json["think"] is True
            assert json["options"]["num_predict"] == -1
            return Response()

    monkeypatch.setattr(capture.httpx, "Client", Client)
    if timeout:
        with pytest.raises(SystemExit) as result:
            capture.main()
        assert result.value.code == 1
    else:
        capture.main()

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[1]["record"]["message"]["thinking"] == "checking evidence"
    assert records[-1]["status"] == ("inactivity_timeout" if timeout else "done")
    assert records[-1]["answer_chars"] == (0 if timeout else len("answer"))
