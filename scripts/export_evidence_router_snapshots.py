#!/usr/bin/env python3
"""Export validated, source-derived evidence-router results for the benchmark UI.

This reads saved local benchmark artifacts and the pinned catalog; it makes no
model or network calls. For each dataset it checks the source and baseline
hashes, case/label membership, exact classifier prompt hashes, saved summaries,
and each clipped search passage against the catalog and recorded index identity.
Only the explicitly permitted fields are exported; host data is not included.

Run from the repository root: `uv run --locked python scripts/export_evidence_router_snapshots.py`.
Use --run-dir and --output-dir to choose alternative input/output directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))
from app.corpus import load_catalog, render_chunks
from app.decision_router import ROUTE_PROMPT
from app.graph import SchemaGraph
from app.index import EMBEDDING_FRAGMENT_CHARS
from scripts.benchmark_evidence_router import evidence_messages, initial_query
from scripts.benchmark_router_decisions import accepted_routes, summarize, validate_dataset


TASKS = (
    ("routing-evidence-aware-test.json", "routing-decisions-test-small.json",
     "routing-decision-v1.json", "routing-evidence-aware-snapshot.json"),
    ("routing-scope-evidence-aware-test.json", "routing-scope-binary-test-small.json",
     "routing-scope-binary-v1.json", "routing-scope-evidence-aware-snapshot.json"),
)
ROW_FIELDS = (
    "model", "case_id", "question", "context_messages", "expected_route", "accepted_routes",
    "initial_query", "initial_hits", "baseline_route", "baseline_status", "baseline_correct",
    "baseline_seconds", "messages_sha256", "status", "predicted_route", "correct", "error",
    "search_seconds", "router_seconds", "end_to_end_seconds", "elapsed_seconds", "runtime",
    "search_runtime", "native_timing",
)
MODEL_FIELDS = ("name", "model", "size", "digest", "details", "capabilities")
MODEL_DETAIL_FIELDS = ("parent_model", "format", "family", "families", "parameter_size", "quantization_level",
                       "context_length", "embedding_length", "runner")
RUNTIME_FIELDS = ("runner", "size_bytes", "gpu_bytes", "context_length")
TIMING_FIELDS = ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration",
                 "eval_count", "eval_duration")
HIT_FIELDS = ("rank", "id", "title", "text", "uri", "score")
SUMMARY_FIELDS = ("attempted", "scored", "errors", "correct", "accuracy", "search_gate_correct",
                  "search_gate_accuracy", "false_search", "false_no_search", "three_way_evaluable",
                  "macro_f1", "mean_seconds", "classes", "mean_end_to_end_seconds",
                  "mean_search_seconds", "mean_router_seconds")
CLASS_FIELDS = ("expected", "predicted", "correct", "precision", "recall", "f1")
SEARCH_FIELDS = ("mode", "reranker", "top_k", "snippet_chars", "query_method", "concurrent_inference",
                 "classifier_hit_fields", "embedder_keep_alive", "router_keep_alive", "max_router_weight_bytes")
SETTINGS_FIELDS = ("think", "max_output_tokens", "max_output_tokens_override", "context_cap_override",
                   "temperature", "context_policy")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def whitelist(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if value is None:
        return {}
    require(isinstance(value, dict), "Expected a JSON object")
    return {key: value[key] for key in fields if key in value}


def clean_model(value: dict[str, Any]) -> dict[str, Any]:
    result = whitelist(value, MODEL_FIELDS)
    if result.get("details") is not None:
        result["details"] = whitelist(result["details"], MODEL_DETAIL_FIELDS)
    return result


def clean_summary(value: dict[str, Any]) -> dict[str, Any]:
    result = whitelist(value, SUMMARY_FIELDS)
    if "classes" in result:
        require(isinstance(result["classes"], dict), "Invalid class summaries")
        result["classes"] = {route: whitelist(result["classes"][route], CLASS_FIELDS)
                             for route in ("raw_search", "rewrite_search", "no_search")
                             if route in result["classes"]}
    return result


def catalog_index(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    catalog = load_catalog(ROOT / "seed/catalog.json")
    require(run.get("source_commit") == catalog.get("source_commit"), "Catalog source revision mismatch")
    SchemaGraph(catalog["entities"], catalog["source_commit"], semantics=catalog.get("semantics", {}))
    chunks = render_chunks(catalog)
    embedding = run["embedding"]
    identity = sha(json.dumps({
        "chunks": chunks, "model": embedding["name"], "model_digest": embedding["digest"],
        "format": "cdm-passages-v3", "embedding_fragment_chars": EMBEDDING_FRAGMENT_CHARS,
        "embedding_format": "nomic-search-v1",
    }, ensure_ascii=False, sort_keys=True).encode())
    require(identity == run.get("index_identity"), "Pinned catalog/index identity mismatch")
    result = {chunk["id"]: chunk for chunk in chunks}
    require(len(result) == len(chunks), "Duplicate catalog passage ID")
    return result


def validate_and_sanitize(run: dict[str, Any], raw: bytes, dataset: dict[str, Any],
                          dataset_raw: bytes, baseline: dict[str, Any], baseline_raw: bytes,
                          by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    require(run.get("status") == "completed" and run.get("split") == "test", "Incomplete run")
    require(run.get("dataset_id") == dataset.get("dataset_id"), "Dataset ID mismatch")
    require(run.get("dataset_sha256") == sha(dataset_raw), "Dataset hash mismatch")
    require(run.get("baseline_sha256") == sha(baseline_raw), "Question-only baseline hash mismatch")
    require(baseline.get("dataset_id") == dataset["dataset_id"]
            and baseline.get("dataset_sha256") == sha(dataset_raw)
            and baseline.get("prompt_sha256") == sha(ROUTE_PROMPT.encode()), "Baseline identity mismatch")
    require(run.get("model_settings", {}).get("think") is False
            and run.get("model_settings", {}).get("max_output_tokens") == 160
            and baseline.get("model_settings", {}).get("think") is False
            and baseline.get("model_settings", {}).get("max_output_tokens") == 160,
            "Router settings differ from the saved question-only baseline")
    cases = {c["id"]: c for c in validate_dataset(dataset) if c["split"] == "test"}
    selected = run.get("selected_case_ids")
    require(isinstance(selected, list) and len(selected) == len(set(selected))
            and set(selected) == set(cases), "Expected exactly all test cases")
    models = run.get("models")
    require(isinstance(models, dict) and models, "No model metadata")
    baseline_rows = {(r["model"], r["case_id"]): r for r in baseline.get("rows", [])}
    require(all(models[tag].get("digest") == baseline["models"][tag].get("digest") for tag in models),
            "Router model differs from baseline")
    search = run.get("search", {})
    require(search.get("mode") == "hybrid" and search.get("reranker") is None
            and search.get("top_k") == 3 and search.get("snippet_chars") == 700,
            "Unexpected search configuration")
    seen: set[tuple[str, str]] = set()
    clean_rows = []
    for row in run.get("rows", []):
        key = (row.get("model"), row.get("case_id"))
        require(key[0] in models and key[1] in cases and key not in seen, "Unknown or duplicate result")
        seen.add(key)
        case = cases[key[1]]
        ref = baseline_rows.get(key)
        require(ref is not None and row.get("question") == case["question"]
                and row.get("context_messages", []) == case.get("context_messages", [])
                and row.get("expected_route") == case.get("expected_route")
                and row.get("accepted_routes") == accepted_routes(case), "Case or reference mismatch")
        require(row.get("initial_query") == initial_query(case), "Preliminary query differs from raw conversation")
        for dest, src in (("baseline_route", "predicted_route"), ("baseline_status", "status"),
                          ("baseline_correct", "correct"), ("baseline_seconds", "elapsed_seconds")):
            require(row.get(dest) == ref.get(src), "Question-only decision does not match saved baseline")
        hits = row.get("initial_hits")
        require(isinstance(hits, list) and len(hits) <= search["top_k"], "Invalid preliminary hits")
        for rank, hit in enumerate(hits, 1):
            passage = by_id.get(hit.get("id"))
            require(passage is not None and hit.get("rank") == rank, "Unknown passage or invalid rank")
            require(hit.get("title") == passage.get("title", "") and hit.get("uri") == passage.get("uri", "")
                    and hit.get("text") == passage.get("text", "")[:search["snippet_chars"]],
                    "Saved excerpt differs from pinned catalog")
            score = hit.get("score")
            require(isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score),
                    "Invalid saved search score")
        if row.get("status") == "scored":
            messages = evidence_messages(case, hits)
            digest = sha(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode())
            require(row.get("messages_sha256") == digest, "Saved classifier prompt does not match reconstructed prompt")
            require(row.get("predicted_route") in {"raw_search", "rewrite_search", "no_search"}
                    and row.get("correct") == (row["predicted_route"] in accepted_routes(case)),
                    "Predicted route or score mismatch")
        else:
            require(row.get("status") == "error" and row.get("predicted_route") is None
                    and row.get("correct") is False, "Invalid error result")
        for field in ("search_seconds", "router_seconds", "end_to_end_seconds", "elapsed_seconds"):
            value = row.get(field)
            require(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0, "Invalid timing")
        require(abs(row["elapsed_seconds"] - row["end_to_end_seconds"]) < 1e-6,
                "Elapsed and end-to-end timing mismatch")
        clean = whitelist(row, ROW_FIELDS)
        clean["initial_hits"] = [whitelist(hit, HIT_FIELDS) for hit in hits]
        for field in ("runtime", "search_runtime"):
            if field in clean and clean[field] is not None:
                clean[field] = whitelist(clean[field], RUNTIME_FIELDS)
        if clean.get("native_timing") is not None:
            clean["native_timing"] = whitelist(clean["native_timing"], TIMING_FIELDS)
        require("/Users/" not in json.dumps(clean, ensure_ascii=False), "Private path in output")
        clean_rows.append(clean)
    require(seen == {(tag, case_id) for tag in models for case_id in selected}, "Missing model/case result")
    for tag in models:
        rows = [r for r in run["rows"] if r["model"] == tag]
        computed = summarize(rows)
        summary = run["summaries"][tag]
        require(all(summary.get(k) == v for k, v in computed.items()), "Classifier summary mismatch")
        expected_baseline = summarize([baseline_rows[(tag, case_id)] for case_id in selected])
        # Earlier runs predate additional derived summary fields. Every value
        # they did save must still agree with the current recomputation.
        require(all(expected_baseline.get(k) == v for k, v in run["baseline"]["summaries"][tag].items()),
                "Baseline summary mismatch")
        for source, dest in (("end_to_end_seconds", "mean_end_to_end_seconds"),
                             ("search_seconds", "mean_search_seconds"),
                             ("router_seconds", "mean_router_seconds")):
            require(math.isclose(summary[dest], sum(r[source] for r in rows) / len(rows), abs_tol=1e-10),
                    "Mean timing mismatch")
    return {
        "schema_version": 1, "description": run["method"], "caveat": run["caveat"],
        "artifact_sha256": sha(raw), "dataset_id": run["dataset_id"], "dataset_sha256": run["dataset_sha256"],
        "split": run["split"], "selected_case_ids": selected, "baseline_sha256": run["baseline_sha256"],
        "baseline": {"summaries": {name: clean_summary(summary)
                                    for name, summary in run["baseline"]["summaries"].items() if name in models},
                     "classifier_sha256": run["baseline"]["classifier_sha256"],
                     "prompt_sha256": run["baseline"]["prompt_sha256"]},
        "index_identity": run["index_identity"],
        "source_commit": run["source_commit"], "index_baseline_sha256": run["index_baseline_sha256"],
        "model_settings": whitelist(run["model_settings"], SETTINGS_FIELDS),
        "search": whitelist(run["search"], SEARCH_FIELDS),
        "models": {name: clean_model(info) for name, info in models.items()},
        "embedding": clean_model(run["embedding"]),
        "script_sha256": run["script_sha256"], "classifier_sha256": run["classifier_sha256"],
        "helper_sha256": run["helper_sha256"], "rows": clean_rows,
        "summaries": {name: clean_summary(run["summaries"][name]) for name in models},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/macstudio-runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation/results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for run_name, baseline_name, dataset_name, output_name in TASKS:
        raw = (args.run_dir / run_name).read_bytes()
        baseline_raw = (args.run_dir / baseline_name).read_bytes()
        dataset_raw = (ROOT / "evaluation" / dataset_name).read_bytes()
        run, baseline, dataset = json.loads(raw), json.loads(baseline_raw), json.loads(dataset_raw)
        snapshot = validate_and_sanitize(run, raw, dataset, dataset_raw, baseline, baseline_raw, catalog_index(run))
        snapshots.append((output_name, snapshot))
    primary, scope = snapshots[0][1], snapshots[1][1]
    primary9, primary4 = (primary["summaries"][tag] for tag in ("qwen3.5:9b-q4_K_M", "qwen3:4b"))
    scope9, scope4 = (scope["summaries"][tag] for tag in ("qwen3.5:9b-q4_K_M", "qwen3:4b"))
    if (primary9["correct"] > primary4["correct"] and scope9["correct"] > scope4["correct"]
            and not any(summary["errors"] for summary in (primary9, primary4, scope9, scope4))):
        primary["recommendation"] = {
            "model": "qwen3.5:9b-q4_K_M",
            "reason": (
                "Qwen 3.5 9B is the provisional evidence-aware routing candidate: it scored "
                f"{primary9['correct']}/{primary9['attempted']} on the three-way set and "
                f"{scope9['correct']}/{scope9['attempted']} on the separate binary scope set, versus "
                f"{primary4['correct']}/{primary4['attempted']} and {scope4['correct']}/{scope4['attempted']} "
                f"for 4B, at about {primary9['mean_end_to_end_seconds']:.2f}s vs "
                f"{primary4['mean_end_to_end_seconds']:.2f}s and {scope9['mean_end_to_end_seconds']:.2f}s vs "
                f"{scope4['mean_end_to_end_seconds']:.2f}s end-to-end. The 27B model was not tested with "
                "evidence-aware routing. Small, previously inspected provisional sets and one run do not "
                "establish production readiness."
            ),
        }
    for output_name, snapshot in snapshots:
        output = args.output_dir / output_name
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(output)
        print(f"{output}: {len(snapshot['rows'])} validated rows")


if __name__ == "__main__":
    main()
