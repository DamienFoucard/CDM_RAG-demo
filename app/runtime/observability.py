"""Persistent local JSONL query logging."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger("cdm_rag.query")
DATA_DIR = Path(os.getenv("CDM_RAG_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
QUERY_LOG_PATH = DATA_DIR / "query_logs.jsonl"
_LOG_LOCK = threading.Lock()


def new_query_id() -> str:
    return str(uuid.uuid4())


def _write_event(query_id: str, event: str, **fields: Any) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_id": query_id,
        "event": event,
        **fields,
    }
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
    logger.info("%s", line)
    try:
        with _LOG_LOCK:
            QUERY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with QUERY_LOG_PATH.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
    except OSError:
        logger.exception("query log file write failed query_id=%s", query_id)


class QueryRecorder:
    def __init__(self, query_id: str, question: str, *, mode: str, rerank: bool, limit: int,
                 models: dict[str, str] | None = None):
        self.query_id = query_id
        self.question = question
        self.mode = mode
        self.rerank_requested = rerank
        self.limit = limit
        self.models = models or {}
        self.started_at = time.monotonic()
        self.current_stage: str | None = None
        self.stage_started_at: float | None = None
        self.stage_timings: list[dict[str, Any]] = []

    def start(self) -> None:
        _write_event(
            self.query_id,
            "query_started",
            question=self.question,
            mode=self.mode,
            rerank_requested=self.rerank_requested,
            limit=self.limit,
            models=self.models,
        )

    def stage(self, name: str) -> None:
        self._finish_stage()
        self.current_stage = name
        self.stage_started_at = time.monotonic()
        _write_event(self.query_id, "stage_started", stage=name)

    def generation_input(self, metrics: dict[str, Any]) -> None:
        _write_event(self.query_id, "generation_input_prepared", **metrics)

    def _finish_stage(self, error: BaseException | None = None) -> None:
        if self.current_stage is None or self.stage_started_at is None:
            return
        duration_ms = round((time.monotonic() - self.stage_started_at) * 1000)
        status = "failed" if error else "completed"
        timing: dict[str, Any] = {"stage": self.current_stage, "duration_ms": duration_ms, "status": status}
        if error:
            timing["error_type"] = type(error).__name__
        self.stage_timings.append(timing)
        _write_event(self.query_id, "stage_failed" if error else "stage_completed", **timing)
        self.current_stage = None
        self.stage_started_at = None

    def complete(self, result: dict[str, Any]) -> None:
        self._finish_stage()
        elapsed_ms = round((time.monotonic() - self.started_at) * 1000)
        retrieval = result.get("retrieval", {})
        _write_event(
            self.query_id,
            "query_completed",
            elapsed_ms=elapsed_ms,
            models=self.models,
            route=result.get("route"),
            mode=retrieval.get("mode", self.mode),
            reranked=retrieval.get("reranked", False),
            reranker_model=retrieval.get("reranker_model"),
            generator=retrieval.get("generator"),
            candidate_count=retrieval.get("candidate_count"),
            retrieval={key: retrieval.get(key) for key in (
                "search_query", "scope_category", "router_backend", "router_decision", "router_model",
                "reranker_model", "omitted_passage_ids", "generation_passage_limit", "generation_passage_ids",
                "preflight_similarity", "threshold",
            ) if key in retrieval},
            answer_quality=result.get("answer_quality"),
            citations=[item.get("citation") for item in result.get("sources", [])],
            stage_timings=self.stage_timings,
            answer=result.get("answer"),
        )

    def fail(self, error: BaseException) -> None:
        failed_stage = self.current_stage or (self.stage_timings[-1]["stage"] if self.stage_timings else None)
        self._finish_stage(error)
        elapsed_ms = round((time.monotonic() - self.started_at) * 1000)
        cause: BaseException | None = error
        while cause is not None and cause.__cause__ is not None:
            cause = cause.__cause__
        _write_event(
            self.query_id,
            "query_failed",
            elapsed_ms=elapsed_ms,
            failed_stage=failed_stage,
            error_type=type(error).__name__,
            error_message=str(error),
            cause_type=type(cause).__name__ if cause is not None and cause is not error else None,
            cause_message=str(cause) if cause is not None and cause is not error else None,
            stage_timings=self.stage_timings,
        )
