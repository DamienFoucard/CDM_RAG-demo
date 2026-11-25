#!/usr/bin/env python3
"""Replay claim review on saved answers and their original cited passages.

The human observations are diagnostic annotations. They are never sent to the
review model and do not make the model's output a ground-truth factuality score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.comparison_ollama import ComparisonOllama, add_comparison_options, comparison_settings


# Observations checked against the saved passages, not a comprehensive gold set.
# None of these annotations is included in the input given to the model.
OBSERVATIONS = {
    ("paired-dev-card-customer", "raw"): "Asserts a cardholder/holding relationship without an edge proving it.",
    ("paired-dev-card-customer", "rewrite"): "Implies the cardholder and holding customer are linked without evidence of record identity.",
    ("paired-dev-holding-details", "rewrite"): "Asserts CustomerRelationship connects customer identity to the holding without supporting evidence.",
    ("paired-test-customer-partner", "rewrite"): "Uses a Contact citation to support an Account definition.",
    ("paired-test-card-holding-roles", "rewrite"): "Says financialHoldingRole identifies the cardholder, conflating a role with FICard.Cardholder.",
}


def load_review_function(revision: str | None) -> tuple[Callable[..., dict[str, Any]], str]:
    """Load current code or the exact committed review logic for a baseline."""
    if revision is None:
        from app.retrieval import rag
        content = (ROOT / "app/retrieval/rag.py").read_bytes()
        return rag.review_answer, hashlib.sha256(content).hexdigest()
    source = None
    legacy = False
    for path in ("app/retrieval/rag.py", "app/rag.py"):
        result = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=ROOT,
                                capture_output=True)
        if result.returncode == 0:
            source = result.stdout
            legacy = path == "app/rag.py"
            break
    if source is None:
        raise ValueError(f"No RAG source found at revision {revision!r}")
    executable = source.decode("utf-8")
    if legacy:
        # Old commits used root-relative imports; only resolve those imports to
        # their current modules. Hash the untouched source for reproducibility.
        locations = {"decision_router": "retrieval", "index": "retrieval",
                     "query_anchor": "retrieval", "router": "retrieval",
                     "scope": "retrieval", "ollama": "runtime"}
        executable = re.sub(r"(?m)^(\s*from )\.(\w+)(\s+import\b)",
                            lambda m: m[1] + "app." + locations.get(m[2], "")
                            + ("." if m[2] in locations else "") + m[2] + m[3], executable)
    module = types.ModuleType("app._saved_claim_review")
    module.__package__ = "app"
    exec(compile(executable, f"{revision}:{path}", "exec"), module.__dict__)
    return module.review_answer, hashlib.sha256(source).hexdigest()


def replay(payloads: list[dict[str, Any]], review: Callable[..., dict[str, Any]], models: Any,
           on_row: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    """Review the selected saved raw drafts using their original source labels."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for payload in payloads:
        for case in payload["rows"]:
            for arm_name, arm in case["arms"].items():
                key = (case["id"], arm_name)
                if key not in OBSERVATIONS:
                    continue
                if key in seen:
                    raise ValueError(f"Duplicate saved case and arm: {key}")
                seen.add(key)
                row: dict[str, Any] = {"case_id": key[0], "arm": key[1],
                                       "human_observation": OBSERVATIONS[key]}
                started = time.perf_counter()
                try:
                    if arm.get("status") != "scored":
                        raise ValueError("Saved arm was not scored and cannot be reviewed")
                    result = review(case["question"], arm["raw_answer"], arm["sources"], arm["contexts"], models)
                    row.update(status="reviewed", review=result,
                               accepted_despite_observation=result.get("decision") == "accept")
                except Exception as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                rows.append(row)
                if on_row:
                    on_row(row)
    if not rows:
        raise ValueError("No manually annotated case and arm found in the supplied results")
    return rows


def _write(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temp, 0o600)
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--revision", help="Git revision containing the review logic to replay; default is current code")
    add_comparison_options(parser)
    args = parser.parse_args()
    inputs = [(path, path.read_bytes()) for path in args.input]
    payloads = [json.loads(content) for _, content in inputs]
    review, code_hash = load_review_function(args.revision)
    response = httpx.get(args.base_url.rstrip("/") + "/api/tags", timeout=10, trust_env=False)
    response.raise_for_status()
    available = {item["name"]: item for item in response.json()["models"]}
    if args.judge not in available:
        raise ValueError("Selected judge must be installed under its exact tag")
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write("{}\n")
    results: dict[str, Any] = {
        "status": "running", "review_revision": args.revision or "working tree",
        "review_source_sha256": code_hash, "judge": available[args.judge],
        "implementation_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ("app/runtime/ollama.py", "scripts/comparison_ollama.py", "scripts/review_saved_claims.py")},
        "model_settings": comparison_settings(args.reasoning_effort, args.max_output_tokens, 600, args.context_cap),
        "input_sha256": {str(path): hashlib.sha256(content).hexdigest() for path, content in inputs},
        "notes": "Manual issue annotations are not shown to the model. Rejection is not proof of correctness.",
        "rows": [],
    }
    _write(args.output, results)
    models = ComparisonOllama(args.base_url, "nomic-embed-text:latest", args.judge, args.judge,
                              judge_model=args.judge, reasoning_effort=args.reasoning_effort,
                              max_output_tokens=args.max_output_tokens, context_cap=args.context_cap)

    def save_row(row: dict[str, Any]) -> None:
        results["rows"].append(row)
        print(f"Reviewed {row['case_id']} / {row['arm']}: {row['status']}", flush=True)
        _write(args.output, results)

    try:
        replay(payloads, review, models, save_row)
        results["status"] = "completed_with_errors" if any(row["status"] != "reviewed" for row in results["rows"]) else "completed"
    except Exception as exc:
        results.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            models.unload_if_loaded(args.judge)
        finally:
            _write(args.output, results)


if __name__ == "__main__":
    main()
