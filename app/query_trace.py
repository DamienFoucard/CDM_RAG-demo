"""Track model calls made by one request without changing the shared client."""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable

active_trace: ContextVar[TracedModels | None] = ContextVar('cdm_query_trace', default=None)


def trace_embedding(method: Callable) -> Callable:
    """Capture index-owned embedding calls only in the current request thread."""
    @wraps(method)
    def wrapped(models: Any, *args: Any, **kwargs: Any) -> Any:
        trace = active_trace.get()
        if trace is None:
            return method(models, *args, **kwargs)
        return trace.record('embedding', models.embedding_model, lambda: method(models, *args, **kwargs))
    return wrapped


class TracedModels:
    def __init__(self, models: Any, stage: Callable[[], str | None],
                 on_call: Callable[[dict[str, str]], None] | None = None):
        self._models = models
        self._stage = stage
        self._on_call = on_call
        self.calls: list[dict[str, str]] = []

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._models, name)
        if name not in {"chat_json", "chat_json_timed", "chat_stream"}:
            return value

        def call(model: str, *args: Any, **kwargs: Any) -> Any:
            stage = self._stage() or "unknown"
            role = {
                "checking_scope": "scope",
                "planning_search": "query" if name == "chat_json" else "routing",
                "reranking": "reranking",
                "writing_answer": "answer",
                "retrying_answer": "answer",
                "reviewing_answer": "judge",
            }.get(stage, "other")
            return self.record(role, model, lambda: value(model, *args, **kwargs))

        return call

    def record(self, role: str, model: str, operation: Callable[[], Any]) -> Any:
        entry = {"role": role, "model": model, "stage": self._stage() or "unknown", "outcome": "started"}
        self.calls.append(entry)
        if self._on_call:
            self._on_call(dict(entry))
        try:
            result = operation()
        except Exception:
            entry["outcome"] = "failed"
            raise
        entry["outcome"] = "completed"
        return result
