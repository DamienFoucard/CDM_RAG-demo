#!/usr/bin/env python3
"""Replay manually supported saved drafts against one or more local reviewers.

The saved stream contains retrieved passages, but not the original graph context
objects. Therefore this replays with the exact saved passages and empty contexts,
not the full original context. Expected decisions are never sent to reviewers.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.comparison_ollama import ComparisonOllama, add_comparison_options, comparison_settings
from scripts.review_saved_claims import load_review_function

CONTROLS = (
    ("gemma4:12b", "bank-01"),
    ("qwen3.5:9b-q4_K_M", "bank-01"),
    ("gemma4:12b", "bank-03"),
    ("qwen3.5:9b-q4_K_M", "test-10"),
)


def collect_controls(raw: bytes) -> list[dict[str, Any]]:
    events = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: ")]
    final = [event["result"] for event in events if isinstance(event.get("result"), dict)]
    if len(final) != 1:
        raise ValueError("Expected exactly one final result in the saved SSE stream")
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for row in final[0]["rows"]:
        key = (row.get("model"), row.get("question_id"))
        if key not in CONTROLS:
            continue
        if key in found:
            raise ValueError(f"Duplicate control {key}")
        if row.get("error") or not row.get("raw_answer") or not row.get("retrieved_sources"):
            raise ValueError(f"Control is missing its answer or retrieved passages: {key}")
        found[key] = {"question": row["question"], "answer": row["raw_answer"],
                      "sources": row["retrieved_sources"], "contexts": []}
    if set(found) != set(CONTROLS):
        raise ValueError(f"Missing controls: {set(CONTROLS) - set(found)}")
    return [{"draft_model": model, "question_id": question_id, **found[(model, question_id)]}
            for model, question_id in CONTROLS]


def replay(controls: list[dict[str, Any]], reviewer: Callable[..., dict[str, Any]], models: Any,
           on_row: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    rows = []
    for control in controls:
        arguments = copy.deepcopy({key: control[key] for key in ("question", "answer", "sources", "contexts")})
        digest = hashlib.sha256(json.dumps(arguments, sort_keys=True, ensure_ascii=False,
                                           separators=(",", ":")).encode()).hexdigest()
        row = {"draft_model": control["draft_model"], "question_id": control["question_id"],
               "control_sha256": digest}
        start = time.perf_counter()
        try:
            result = reviewer(arguments["question"], arguments["answer"], arguments["sources"],
                              arguments["contexts"], models)
            row.update(status="reviewed", review=result, rejected_supported_draft=result.get("decision") != "accept")
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        row["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        rows.append(row)
        if on_row:
            on_row(row)
    return rows


def write_result(path: Path, data: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge", action="append", required=True, help="Installed reviewer tag; repeat to compare")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--revision", help="Git revision containing reviewer code; default is working tree")
    add_comparison_options(parser)
    args = parser.parse_args()
    if len(set(args.judge)) != len(args.judge):
        parser.error("Reviewer tags must be unique")
    raw = args.input.read_bytes()
    controls = collect_controls(raw)
    review, code_hash = load_review_function(args.revision)
    response = httpx.get(args.base_url.rstrip("/") + "/api/tags", timeout=10, trust_env=False)
    response.raise_for_status()
    available = {model["name"]: model for model in response.json()["models"]}
    if any(tag not in available for tag in args.judge):
        raise ValueError("Every reviewer must be installed under its exact tag")
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write("{}\n")
    data: dict[str, Any] = {"status": "running", "review_revision": args.revision or "working tree",
                            "review_source_sha256": code_hash, "input_sha256": hashlib.sha256(raw).hexdigest(),
                            "implementation_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("app/ollama.py", "scripts/comparison_ollama.py", "scripts/replay_reviewer_positive_controls.py")},
                            "model_settings": comparison_settings(args.reasoning_effort, args.max_output_tokens, 600, args.context_cap),
                            "note": "Source-checked controls; contexts unavailable in saved SSE, so replay uses empty contexts. Two controls share a question and source. Expected outcome never sent to model.",
                            "judges": []}
    write_result(args.output, data)
    try:
        for tag in args.judge:
            group = {"judge": available[tag], "rows": []}
            data["judges"].append(group)
            models = ComparisonOllama(args.base_url, "nomic-embed-text:latest", tag, tag, judge_model=tag,
                                      reasoning_effort=args.reasoning_effort, max_output_tokens=args.max_output_tokens,
                                      context_cap=args.context_cap)
            def save(row: dict[str, Any]) -> None:
                group["rows"].append(row)
                write_result(args.output, data)
                print(f"{tag}: {row['question_id']} / {row['draft_model']}: {row['status']}", flush=True)
            try:
                replay(controls, review, models, save)
            finally:
                models.unload_if_loaded(tag)
        data["status"] = ("completed_with_errors" if any(row["status"] != "reviewed" for group in data["judges"]
                                                        for row in group["rows"]) else "completed")
    except Exception as exc:
        data.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_result(args.output, data)


if __name__ == "__main__":
    main()
