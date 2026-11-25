#!/usr/bin/env python3
"""Compare answer models on exact evidence from saved paired runs.

No retrieval, rewriting or automatic judging is performed. Run as an isolated
model batch; this script does not acquire a host model-work lease or the app lock.
"""
from __future__ import annotations

import argparse
import copy
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

from app.retrieval.rag import cited_answer, format_entity_citations
from scripts.comparison_ollama import ComparisonOllama, add_comparison_options, comparison_settings


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def collect_cases(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep each saved question and arm's exact source and context objects."""
    cases = []
    seen: set[tuple[str, str]] = set()
    for payload in payloads:
        for row in payload.get("rows", []):
            for arm_name in ("raw", "rewrite"):
                arm = row.get("arms", {}).get(arm_name, {})
                key = (row.get("id"), arm_name)
                if not isinstance(key[0], str) or key in seen:
                    raise ValueError("Saved case IDs must be unique")
                if arm.get("status") != "scored":
                    raise ValueError(f"Saved arm is not scored: {key}")
                if not isinstance(arm.get("contexts"), list) or not isinstance(arm.get("sources"), list):
                    raise ValueError(f"Saved evidence is missing: {key}")
                if not arm["contexts"] or not arm["sources"]:
                    raise ValueError(f"Saved evidence is empty: {key}")
                seen.add(key)
                cases.append({"case_id": key[0], "arm": arm_name,
                              "question": row["question"], "reference_answer": row.get("reference_answer"),
                              "query": arm.get("query"),
                              "contexts": copy.deepcopy(arm["contexts"]),
                              "sources": copy.deepcopy(arm["sources"]),
                              "evidence_sha256": content_hash({"contexts": arm["contexts"],
                                                               "sources": arm["sources"]})})
    if not cases:
        raise ValueError("No saved paired evidence found")
    return cases


def select_cases(cases: list[dict[str, Any]], case_ids: list[str] | None,
                 arms: list[str] | None = None) -> list[dict[str, Any]]:
    """Select saved cases and arms without changing input order."""
    if case_ids is not None:
        if not case_ids or len(case_ids) != len(set(case_ids)):
            raise ValueError("Case IDs must be nonempty and unique")
        available = {case["case_id"] for case in cases}
        if set(case_ids) - available:
            raise ValueError("Every selected case ID must exist in the saved inputs")
    if arms is not None:
        if not arms or len(arms) != len(set(arms)) or set(arms) - {"raw", "rewrite"}:
            raise ValueError("Arms must be nonempty, unique, and either raw or rewrite")
    return [case for case in cases if (case_ids is None or case["case_id"] in case_ids)
            and (arms is None or case["arm"] in arms)]


def generate_case(case: dict[str, Any], models: Any) -> dict[str, Any]:
    """Generate with a private copy; never replace the fixed saved evidence."""
    result = {"case_id": case["case_id"], "arm": case["arm"],
              "model": models.answer_model, "evidence_sha256": case["evidence_sha256"]}
    started = time.perf_counter()
    # The client stores native timing counters in thread-local state even if
    # the model returns an empty answer. Clear a prior case before starting so
    # an exception before the model call cannot inherit stale counters.
    chat_local = getattr(models, "_chat_local", None)
    if chat_local is not None:
        chat_local.timings = None
    try:
        sources = copy.deepcopy(case["sources"])
        contexts = copy.deepcopy(case["contexts"])
        raw, generator = cited_answer(case["question"], sources, contexts, models)
        answer, citations = format_entity_citations(raw, sources)
        result.update(status="fallback" if generator in {"none", "extractive-fallback", "error"} else "generated",
                      generator=generator, raw_answer=raw, answer=answer, cited_sources=citations,
                      runtime=models.runtime_info(models.answer_model))
    except Exception as exc:
        timings = getattr(chat_local, "timings", None)
        result.update(status="error", error=f"{type(exc).__name__}: {exc}",
                      error_type=type(exc).__name__,
                      native_timing=copy.deepcopy(timings) if isinstance(timings, dict) else None)
    result["generation_seconds"] = round(time.perf_counter() - started, 3)
    return result


def write_result(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--model", action="append", required=True,
                        help="Exact installed model tag; repeat for each generator")
    parser.add_argument("--case-id", action="append",
                        help="Saved question ID to include; repeat for multiple cases")
    parser.add_argument("--arm", action="append", choices=("raw", "rewrite"),
                        help="Saved retrieval arm to include; repeat for both (default: both)")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path, required=True)
    add_comparison_options(parser)
    args = parser.parse_args()
    if len(args.model) != len(set(args.model)):
        raise ValueError("Model tags must be unique")
    inputs = [(path, path.read_bytes()) for path in args.input]
    cases = select_cases(collect_cases([json.loads(content) for _, content in inputs]), args.case_id, args.arm)
    response = httpx.get(args.base_url.rstrip("/") + "/api/tags", timeout=10, trust_env=False)
    response.raise_for_status()
    available = {item["name"]: item for item in response.json()["models"]}
    if any(name not in available for name in args.model):
        raise ValueError("Every model must be installed under its exact tag")
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write("{}\n")
    relevant = ("app/retrieval/rag.py", "app/runtime/ollama.py", "app/retrieval/query_anchor.py", "scripts/comparison_ollama.py")
    result: dict[str, Any] = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "method": "Saved question, source labels and passage text held fixed; regenerate each arm per model; no judge.",
        "notes": "A valid citation label does not establish claim support; manually assess answers against saved passages.",
        "input_sha256": {str(path): hashlib.sha256(content).hexdigest() for path, content in inputs},
        "implementation_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in relevant},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_settings": comparison_settings(args.reasoning_effort, args.max_output_tokens, 700, args.context_cap),
        "selected_case_ids": list(dict.fromkeys(case["case_id"] for case in cases)),
        "selected_arms": list(dict.fromkeys(case["arm"] for case in cases)),
        "host": {"system": platform.platform(), "machine": platform.machine()},
        "models": {name: available[name] for name in args.model},
        "cases": cases, "rows": [],
    }
    write_result(args.output, result)
    try:
        for name in args.model:
            # A private, unused judge tag avoids model-name collisions in the
            # role-based context policy; both generators use the answer policy.
            models = ComparisonOllama(args.base_url, "nomic-embed-text:latest", "__unused_router__", name,
                                      judge_model="__unused_judge__", reasoning_effort=args.reasoning_effort,
                                      max_output_tokens=args.max_output_tokens, context_cap=args.context_cap)
            try:
                for case in cases:
                    print(f"Generating {name}: {case['case_id']} / {case['arm']}", flush=True)
                    result["rows"].append(generate_case(case, models))
                    write_result(args.output, result)
            finally:
                models.unload_if_loaded(name)
        result["status"] = ("completed_with_errors" if any(row["status"] != "generated" for row in result["rows"])
                            else "completed")
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_result(args.output, result)


if __name__ == "__main__":
    main()
