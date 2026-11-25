"""Reasoning counts as stream activity, but never leaks into an answer."""
import json
import time

import pytest

from app.ollama import ModelError, Ollama


def _mock_stream(monkeypatch, events):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self):
            for delay, message in events:
                time.sleep(delay)
                yield json.dumps({"message": message})

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def stream(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr("app.ollama.httpx.Client", Client)


def _model(first=0.3, idle=0.3):
    return Ollama("http://localhost:11434", "embed", "router", "answer",
                  first_token_timeout=first, stream_idle_timeout=idle)


def test_reasoning_activity_keeps_stream_alive_without_becoming_answer(monkeypatch):
    # Overall delay exceeds the first-token limit; every reasoning gap is
    # smaller than the idle limit. The old content-only clock times out.
    _mock_stream(monkeypatch, [(0.13, {"thinking": "private thought"}),
                               (0.13, {"thinking": " more"}),
                               (0.13, {"thinking": " thinking"}),
                               (0.13, {"content": "final answer"})])
    delivered = []
    result = _model().chat_stream("answer", [], on_token=delivered.append)
    assert result == "final answer"
    assert delivered == ["final answer"]


def test_silent_stream_after_reasoning_hits_idle_timeout(monkeypatch):
    _mock_stream(monkeypatch, [(0.01, {"thinking": "active"}),
                               (0.25, {"content": "too late"})])
    with pytest.raises(ModelError, match="produced no output"):
        _model(first=0.4, idle=0.1).chat("answer", [])


def test_reasoning_only_does_not_count_as_final_answer(monkeypatch):
    _mock_stream(monkeypatch, [(0, {"thinking": "private thought"})])
    delivered = []
    with pytest.raises(ModelError, match="empty answer"):
        _model().chat_stream("answer", [], on_token=delivered.append)
    assert delivered == []


def test_empty_thinking_does_not_count_as_activity(monkeypatch):
    _mock_stream(monkeypatch, [(0.08, {"thinking": ""}),
                               (0.08, {"thinking": ""}),
                               (0.08, {"content": "too late"})])
    with pytest.raises(ModelError, match="No first token"):
        _model(first=0.2, idle=0.2).chat("answer", [])
