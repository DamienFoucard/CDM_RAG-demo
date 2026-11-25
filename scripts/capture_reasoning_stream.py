#!/usr/bin/env python3
"""Capture a fixed-evidence answer stream, including model reasoning, locally."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag import cited_answer  # noqa: E402


class PromptCaptured(Exception):
    """Stop before inference after the application constructs its messages."""


class PromptCapture:
    def __init__(self, model: str) -> None:
        self.answer_model = model
        self.messages: list[dict] | None = None

    def chat_stream(self, model: str, messages: list[dict], **kwargs: object) -> None:
        self.messages = copy.deepcopy(messages)
        raise PromptCaptured


def answer_messages(case: dict, model: str) -> list[dict]:
    capture = PromptCapture(model)
    try:
        cited_answer(case["question"], copy.deepcopy(case["sources"]),
                     copy.deepcopy(case["contexts"]), capture)
    except PromptCaptured:
        pass
    if capture.messages is None:
        raise ValueError("The case did not produce an answer prompt; check its saved evidence.")
    return capture.messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Saved fixed-evidence run JSON")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", required=True, help="Installed Ollama model tag")
    parser.add_argument("--output", type=Path, required=True, help="New local NDJSON file")
    parser.add_argument("--context", type=int, default=32768, help="Requested context size")
    parser.add_argument("--idle-timeout", type=float, default=300,
                        help="Seconds without stream data before stopping")
    parser.add_argument("--url", default="http://127.0.0.1:11434", help="Ollama base URL")
    args = parser.parse_args()
    if args.context <= 0 or args.idle_timeout <= 0:
        parser.error("--context and --idle-timeout must be positive")

    cases = json.loads(args.input.read_text(encoding="utf-8"))["cases"]
    matches = [case for case in cases if case.get("case_id") == args.case_id
               and case.get("arm") == args.arm]
    if len(matches) != 1:
        parser.error(f"Expected one matching case and arm; found {len(matches)}")
    case = matches[0]
    messages = answer_messages(case, args.model)
    options = {"temperature": 0, "num_predict": -1, "num_ctx": args.context}
    payload = {"model": args.model, "messages": messages, "stream": True,
               "think": True, "options": options, "keep_alive": 0}
    start = time.monotonic()
    metadata = {
        "case_id": args.case_id, "arm": args.arm,
        "evidence_sha256": case.get("evidence_sha256"), "source_run": str(args.input),
        "model": args.model, "options": options,
        "prompt_sha256": hashlib.sha256(json.dumps(
            messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "method": "Saved question and evidence; current application prompt; no fixed output-token limit.",
    }
    # Saved outputs may contain private questions. Refuse to replace earlier runs.
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        def save(record: dict) -> None:
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            out.flush()

        save({"type": "start", **metadata})
        count = thinking_chars = answer_chars = 0
        finished = False
        status = "stream_ended_without_done"
        try:
            timeout = httpx.Timeout(connect=10, read=args.idle_timeout, write=30, pool=10)
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                with client.stream("POST", args.url.rstrip("/") + "/api/chat", json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        record = json.loads(line)
                        elapsed = round(time.monotonic() - start, 3)
                        save({"type": "chunk", "elapsed_seconds": elapsed, "record": record})
                        count += 1
                        message = record.get("message") or {}
                        thinking_chars += len(message.get("thinking") or "")
                        answer_chars += len(message.get("content") or "")
                        if count % 1000 == 0:
                            print(f"{elapsed:.0f}s: {count} chunks, {thinking_chars} reasoning chars, "
                                  f"{answer_chars} answer chars", flush=True)
                        if record.get("error"):
                            status = "server_error"
                            save({"type": "result", "status": "server_error", "error": record["error"],
                                  "elapsed_seconds": elapsed})
                            finished = True
                            break
                        if record.get("done"):
                            status = "done"
                            save({"type": "result", "status": "done", "done_reason": record.get("done_reason"),
                                  "elapsed_seconds": elapsed, "chunks": count,
                                  "thinking_chars": thinking_chars, "answer_chars": answer_chars})
                            finished = True
                            break
            if not finished:
                save({"type": "result", "status": "stream_ended_without_done",
                      "elapsed_seconds": round(time.monotonic() - start, 3)})
        except httpx.ReadTimeout:
            status = "inactivity_timeout"
            save({"type": "result", "status": "inactivity_timeout", "inactivity_seconds": args.idle_timeout,
                  "elapsed_seconds": round(time.monotonic() - start, 3), "chunks": count,
                  "thinking_chars": thinking_chars, "answer_chars": answer_chars})
            print(f"Stopped: no stream data for {args.idle_timeout:g} seconds.", flush=True)
        except Exception as exc:
            save({"type": "result", "status": "error", "error_type": type(exc).__name__,
                  "message": str(exc), "elapsed_seconds": round(time.monotonic() - start, 3),
                  "chunks": count, "thinking_chars": thinking_chars, "answer_chars": answer_chars})
            raise
    print(f"Finished after {time.monotonic() - start:.1f}s; saved {args.output}")
    if status != "done":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
