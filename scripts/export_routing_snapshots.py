#!/usr/bin/env python3
"""Combine local routing-only runs into portable, source-checked page data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "macstudio-runs"
OUTPUT = ROOT / "evaluation" / "results"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combine(dataset_name: str, run_names: list[str], output_name: str) -> None:
    dataset_path = ROOT / "evaluation" / dataset_name
    dataset = json.loads(dataset_path.read_text())
    cases = {case["id"]: case for case in dataset["questions"] if case["split"] == "test"}
    runs = [(RUNS / name, json.loads((RUNS / name).read_text())) for name in run_names]
    result = {"status": "completed", "dataset_id": dataset["dataset_id"],
              "dataset_sha256": digest(dataset_path), "split": "test", "models": {},
              "summaries": {}, "rows": [], "provenance": [],
              "method": "LLM decision only; no rewrite, retrieval or answer generation.",
              "caveat": dataset.get("status", "Provisional labels, not independently reviewed."),
              "classifier_revision_note": "One revision adds an unused optional model argument and live adapter; recorded benchmark prompt, schema, messages, and model selection are unchanged."}
    expected_prompt = runs[0][1]["prompt_sha256"]
    # The second revision adds the live adapter and an optional model argument;
    # these runs omit that argument and use the same classifier/prompt behavior.
    approved_classifiers = {
        "b3f020e34f83692fef8e3adea948bc4bf2fa4f6a5aa19f64632b6233c5186388",
        "482e5a3df68aadaf6a62b34ecfcdf142e98750c4321fb3e49c6d4a68ab537ec4",
    }
    for path, run in runs:
        if (run["status"] != "completed" or run["dataset_id"] != dataset["dataset_id"]
                or run["dataset_sha256"] != result["dataset_sha256"] or run["split"] != "test"
                or run["prompt_sha256"] != expected_prompt or run["classifier_sha256"] not in approved_classifiers):
            raise ValueError(f"Incompatible routing run: {path.name}")
        if set(result["models"]) & set(run["models"]):
            raise ValueError(f"Duplicate model in {path.name}")
        for tag, metadata in run["models"].items():
            result["models"][tag] = {key: metadata[key] for key in ("name", "digest", "size") if key in metadata}
            model_rows = [row for row in run["rows"] if row["model"] == tag]
            if len(model_rows) != len(cases) or {row["case_id"] for row in model_rows} != set(cases):
                raise ValueError(f"Incomplete cases for {tag}")
            for row in model_rows:
                case = cases[row["case_id"]]
                if row["question"] != case["question"] or row.get("context_messages", []) != case.get("context_messages", []):
                    raise ValueError(f"Mismatched case {row['case_id']}")
                result["rows"].append({key: row[key] for key in ("model", "case_id", "question", "context_messages",
                    "expected_route", "accepted_routes", "status", "predicted_route", "correct", "elapsed_seconds", "error") if key in row})
            result["summaries"][tag] = run["summaries"][tag]
        result["provenance"].append({"artifact": path.name, "sha256": digest(path),
            "script_sha256": run["script_sha256"], "prompt_sha256": run["prompt_sha256"],
            "classifier_sha256": run["classifier_sha256"], "model_settings": run["model_settings"]})
    path = OUTPUT / output_name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    combine("routing-decision-v1.json", ["routing-decisions-test-small.json", "routing-decisions-test-27b.json"],
            "routing-benchmark-snapshot.json")
    combine("routing-scope-binary-v1.json", ["routing-scope-binary-test-small.json", "routing-scope-binary-test-27b.json"],
            "routing-scope-binary-snapshot.json")
