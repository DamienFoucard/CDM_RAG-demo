#!/usr/bin/env python3
"""Evaluate routing decisions alone, without query generation or retrieval.

The model receives only the question and preceding conversation; reference
labels are read only after it returns. Models run sequentially. This local CLI
does not acquire the application's model-work lock; run it in isolation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.runtime.ollama import ModelError
from app.retrieval.decision_router import ROUTES, ROUTE_PROMPT, classify_route_timed, route_messages
from scripts.comparison_ollama import ComparisonOllama, add_comparison_options, comparison_settings


def accepted_routes(case: dict[str, Any]) -> list[str]:
    """Allow binary scope labels without inventing a raw-versus-rewrite preference."""
    accepted = case.get("accepted_routes")
    expected = case.get("expected_route")
    if accepted is None:
        accepted = [expected]
    if (not isinstance(accepted, list) or not accepted
            or any(not isinstance(route, str) or route not in ROUTES for route in accepted)
            or len(accepted) != len(set(accepted))
            or ("no_search" in accepted and len(accepted) > 1)
            or (expected is not None and expected not in accepted)):
        raise ValueError("Invalid accepted routing labels")
    return accepted


def validate_dataset(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict) or not isinstance(data.get("dataset_id"), str):
        raise ValueError("A routing dataset with an ID is required")
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("The routing dataset contains no questions")
    seen: set[str] = set()
    for case in questions:
        if (not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]
                or case["id"] in seen or case.get("split") not in {"development", "test"}
                or not isinstance(case.get("question"), str) or not case["question"].strip()):
            raise ValueError("Invalid or duplicate routing case")
        accepted_routes(case)
        seen.add(case["id"])
        messages = case.get("context_messages", [])
        if (not isinstance(messages, list) or any(not isinstance(m, dict)
                or m.get("role") not in {"user", "assistant"}
                or not isinstance(m.get("content"), str) for m in messages)):
            raise ValueError("Invalid prior conversation in routing case")
    return questions


def messages_for(case: dict[str, Any]) -> list[dict[str, str]]:
    return route_messages(case["question"], case.get("context_messages"))


def decision_for(case: dict[str, Any], model: Any) -> dict[str, Any]:
    started = time.perf_counter()
    accepted = accepted_routes(case)
    row = {"case_id": case["id"], "question": case["question"],
           "context_messages": case.get("context_messages", []), "expected_route": case.get("expected_route"),
           "accepted_routes": accepted}
    try:
        decision, native_timing = classify_route_timed(case["question"], model, case.get("context_messages"))
        row.update(status="scored", predicted_route=decision,
                   correct=decision in accepted, native_timing=native_timing,
                   runtime=model.runtime_info(model.router_model))
    except (ModelError, ValueError) as exc:
        row.update(status="error", predicted_route=None, correct=False,
                   error=f"{type(exc).__name__}: {exc}")
    row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    # A scope-only dataset allows both search routes. Such rows cannot supply a
    # raw-vs-rewrite ground truth; don't manufacture a three-class macro score.
    singleton = [r for r in rows if len(accepted_routes(r)) == 1]
    classes = {}
    for route in ROUTES:
        expected = sum(accepted_routes(r) == [route] for r in singleton)
        predicted = sum(r.get("predicted_route") == route for r in singleton)
        true = sum(accepted_routes(r) == [route] and r.get("predicted_route") == route for r in singleton)
        precision = true / predicted if predicted else 0.0
        recall = true / expected if expected else 0.0
        classes[route] = {"expected": expected, "predicted": predicted, "correct": true,
                          "precision": precision, "recall": recall,
                          "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    represented = [v["f1"] for v in classes.values() if v["expected"]]
    gate_correct = sum(r.get("predicted_route") in ROUTES
                       and ((accepted_routes(r) == ["no_search"]) == (r["predicted_route"] == "no_search"))
                       for r in rows)
    return {"attempted": total, "scored": sum(r["status"] == "scored" for r in rows),
            "errors": sum(r["status"] != "scored" for r in rows),
            "correct": sum(bool(r.get("correct")) for r in rows),
            "accuracy": sum(bool(r.get("correct")) for r in rows) / total if total else 0.0,
            "search_gate_correct": gate_correct, "search_gate_accuracy": gate_correct / total if total else 0.0,
            "false_search": sum(accepted_routes(r) == ["no_search"] and r.get("predicted_route") in {"raw_search", "rewrite_search"} for r in rows),
            "false_no_search": sum(accepted_routes(r) != ["no_search"] and r.get("predicted_route") == "no_search" for r in rows),
            "three_way_evaluable": len(singleton),
            "macro_f1": (sum(represented) / len(represented) if represented and len(singleton) == total else None),
            "mean_seconds": sum(r["elapsed_seconds"] for r in rows) / total if total else 0.0,
            "classes": classes}


def save(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/routing-decision-v1.json")
    parser.add_argument("--split", choices=("development", "test"), default="test")
    parser.add_argument("--case-id", action="append", help="Limit to selected IDs in the chosen split")
    parser.add_argument("--model", action="append", required=True, help="Installed local model tag; repeat")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path, required=True)
    add_comparison_options(parser)
    args = parser.parse_args()
    if len(args.model) != len(set(args.model)):
        parser.error("Model tags must be distinct")
    raw = args.dataset.read_bytes()
    dataset = json.loads(raw)
    cases = [case for case in validate_dataset(dataset) if case["split"] == args.split]
    if args.case_id:
        if len(args.case_id) != len(set(args.case_id)) or not set(args.case_id) <= {c["id"] for c in cases}:
            parser.error("Case IDs must be distinct and belong to the chosen split")
        cases = [case for case in cases if case["id"] in args.case_id]
    if not cases:
        parser.error("No cases selected")
    response = httpx.get(args.base_url.rstrip("/") + "/api/tags", timeout=10, trust_env=False)
    response.raise_for_status()
    available = {model["name"]: model for model in response.json()["models"]}
    if any(tag not in available for tag in args.model):
        raise ValueError("All model tags must be installed exactly as selected")
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    result: dict[str, Any] = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "method": "Independent three-way decision only; no rewrite, retrieval or answer generation.",
        "caveat": "Labels are provisional, source-derived and not independently reviewed. Rewrite need is subjective; splits may share source families.",
        "dataset_id": dataset["dataset_id"], "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_status": dataset.get("status"), "dataset_provenance": dataset.get("provenance"),
        "source_commit": dataset.get("source_commit"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256((ROOT / "scripts/comparison_ollama.py").read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(ROUTE_PROMPT.encode()).hexdigest(),
        "classifier_sha256": hashlib.sha256((ROOT / "app/retrieval/decision_router.py").read_bytes()).hexdigest(),
        "split": args.split, "selected_case_ids": [case["id"] for case in cases],
        "model_settings": comparison_settings(args.reasoning_effort, args.max_output_tokens, 160, args.context_cap),
        "host": {"system": platform.platform(), "machine": platform.machine()},
        "models": {tag: available[tag] for tag in args.model}, "rows": [], "summaries": {},
    }
    save(args.output, result)
    try:
        for tag in args.model:
            model = ComparisonOllama(args.base_url, "nomic-embed-text:latest", tag, "__unused_answer__",
                                     judge_model="__unused_judge__", reasoning_effort=args.reasoning_effort,
                                     max_output_tokens=args.max_output_tokens, context_cap=args.context_cap)
            model_rows = []
            try:
                for case in cases:
                    print(f"{tag}: {case['id']}", flush=True)
                    row = {"model": tag, **decision_for(case, model)}
                    model_rows.append(row)
                    result["rows"].append(row)
                    result["summaries"][tag] = summarize(model_rows)
                    save(args.output, result)
            finally:
                model.unload_if_loaded(tag)
        result["status"] = "completed_with_errors" if any(r["status"] != "scored" for r in result["rows"]) else "completed"
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        save(args.output, result)


if __name__ == "__main__":
    main()
