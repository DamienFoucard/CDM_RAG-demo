#!/usr/bin/env python3
"""Compare installed local models using the production CDM scope classifier."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.benchmark import available_chat_models
from app.ollama import ModelError, Ollama
from app.rag import plan_search_timed
from app.scope import SCOPE_CATEGORIES, SCOPE_PROMPT, assess_scope_timed


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def model_digest(base_url: str, name: str) -> str | None:
    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            response = client.get(base_url.rstrip("/") + "/api/tags")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return None
    requested = name.removesuffix(":latest")
    item = next((entry for entry in models if isinstance(entry, dict)
                 and str(entry.get("name", "")).removesuffix(":latest") == requested), None)
    digest = item.get("digest") if isinstance(item, dict) else None
    return digest if isinstance(digest, str) else None


def native_ms(timings: dict[str, int] | None, field: str) -> float | None:
    value = timings.get(field) if isinstance(timings, dict) else None
    return round(value / 1_000_000, 2) if isinstance(value, int) else None


def summarize(rows: list[dict], categories: list[str]) -> dict:
    valid_rows = [row for row in rows if row["predicted_scope"] in categories]
    correct = sum(row["predicted_scope"] == row["expected"] for row in valid_rows)
    schema_rows = [row for row in rows if row["expected"] == "schema"]
    non_schema_rows = [row for row in rows if row["expected"] != "schema"]
    confusion = {expected: {predicted: 0 for predicted in categories + ["error"]}
                 for expected in categories}
    for row in rows:
        prediction = row["predicted_scope"] if row["predicted_scope"] in categories else "error"
        confusion[row["expected"]][prediction] += 1
    planner_rows = [row for row in rows if row["planner_search"] is not None]
    scope_latencies = [row["scope_latency_ms"] for row in rows if row["scope_latency_ms"] is not None]
    planner_latencies = [row["planner_latency_ms"] for row in planner_rows
                         if row["planner_latency_ms"] is not None]
    pipeline_latencies = [row["pipeline_latency_ms"] for row in rows
                          if row["pipeline_latency_ms"] is not None]
    metrics = {
        "cases": len(rows),
        "classification_accuracy": round(correct / len(rows), 4) if rows else None,
        "scope_errors": sum(row["predicted_scope"] not in categories for row in rows),
        "planner_errors": sum(row["planner_error"] is not None for row in rows),
        "search_decision_accuracy": round(sum(
            row["predicted_search"] == (row["expected"] == "schema") for row in rows
        ) / len(rows), 4) if rows else None,
        "schema_cases": len(schema_rows),
        "schema_recall": round(sum(row["predicted_search"] is True for row in schema_rows)
                                / len(schema_rows), 4) if schema_rows else None,
        "non_schema_cases": len(non_schema_rows),
        "non_schema_false_search_rate": round(sum(row["predicted_search"] is True for row in non_schema_rows)
                                               / len(non_schema_rows), 4) if non_schema_rows else None,
        "planner_evaluated_cases": len(planner_rows),
        "planner_search_accuracy": round(sum(row["planner_search"] == row["expected_search"]
                                              for row in planner_rows) / len(planner_rows), 4)
        if planner_rows else None,
        "schema_planner_cases": sum(row["expected_search"] for row in planner_rows),
        "schema_planner_recall": round(sum(row["planner_search"] is True for row in planner_rows
                                             if row["expected_search"])
                                        / sum(row["expected_search"] for row in planner_rows), 4)
        if any(row["expected_search"] for row in planner_rows) else None,
        "query_rewrite_rate": round(sum(row["query_changed"] is True for row in planner_rows)
                                     / len(planner_rows), 4) if planner_rows else None,
        "mean_scope_wall_ms": round(statistics.mean(scope_latencies), 2) if scope_latencies else None,
        "p95_scope_wall_ms": round(percentile(scope_latencies, 0.95), 2) if scope_latencies else None,
        "mean_planner_wall_ms": round(statistics.mean(planner_latencies), 2) if planner_latencies else None,
        "p95_planner_wall_ms": round(percentile(planner_latencies, 0.95), 2) if planner_latencies else None,
        "mean_pipeline_wall_ms": round(statistics.mean(pipeline_latencies), 2) if pipeline_latencies else None,
        "p95_pipeline_wall_ms": round(percentile(pipeline_latencies, 0.95), 2)
        if pipeline_latencies else None,
        "confusion_matrix": confusion,
    }
    for stage in ("scope", "planner"):
        for metric, field in (("mean_load_ms", "ollama_load_ms"),
                              ("mean_prompt_eval_ms", "ollama_prompt_eval_ms"),
                              ("mean_decode_ms", "ollama_decode_ms"),
                              ("decode_tokens_per_second", "ollama_decode_tokens_per_second")):
            values = [row[f"{stage}_{field}"] for row in rows
                      if row[f"{stage}_{field}"] is not None]
            metrics[f"{stage}_{metric}"] = round(statistics.mean(values), 2) if values else None
    return metrics


def load_cases(path: Path, split: str) -> tuple[dict, bytes, list[dict]]:
    raw = path.read_bytes()
    dataset = json.loads(raw)
    cases = dataset.get("cases") if isinstance(dataset, dict) else None
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"{path} does not contain a non-empty cases list.")
    selected = [case for case in cases if isinstance(case, dict)
                and (split == "all" or case.get("split") == split)]
    if not selected:
        raise SystemExit(f"No questions found for split {split!r} in {path}.")
    if any(not isinstance(case.get("id"), str) or not isinstance(case.get("question"), str)
           or case.get("expected") not in SCOPE_CATEGORIES for case in selected):
        raise SystemExit("The dataset contains a case with an invalid ID, question, or scope label.")
    if len({case["id"] for case in selected}) != len(selected):
        raise SystemExit("The selected dataset split contains duplicate question IDs.")
    return dataset, raw, selected


def run_model(models: Ollama, base_url: str, model_info: dict, cases: list[dict],
              split: str, warmup_question: str) -> dict:
    name = model_info["name"]
    models.scope_model = name
    models.router_model = name
    model_digest_value = model_digest(base_url, name)
    models.unload_if_loaded(name)

    cold_started = time.perf_counter()
    warmup_prediction, warmup_native = assess_scope_timed(warmup_question, models)
    cold_start_ms = round((time.perf_counter() - cold_started) * 1000, 2)

    rows = []
    for position, case in enumerate(cases, 1):
        scope_latency = None
        scope_predicted = None
        scope_native = None
        scope_error = None
        scope_started = time.perf_counter()
        try:
            scope_predicted, scope_native = assess_scope_timed(case["question"], models)
        except ModelError as exc:
            scope_error = str(exc)
        scope_latency = round((time.perf_counter() - scope_started) * 1000, 2)

        planner_latency = None
        planner_search = None
        planner_query = None
        planner_native = None
        planner_error = None
        query_changed = None
        if scope_predicted == "schema":
            planner_started = time.perf_counter()
            try:
                planner_search, planner_query, planner_native = plan_search_timed(case["question"], models)
                query_changed = " ".join(planner_query.split()).casefold() != " ".join(
                    case["question"].split()).casefold()
            except ModelError as exc:
                planner_error = str(exc)
                planner_search = False
            planner_latency = round((time.perf_counter() - planner_started) * 1000, 2)

        scope_eval_count = (scope_native.get("eval_count") if isinstance(scope_native, dict) else None)
        scope_eval_duration = (scope_native.get("eval_duration") if isinstance(scope_native, dict) else None)
        scope_decode_rate = (scope_eval_count * 1_000_000_000 / scope_eval_duration
                             if isinstance(scope_eval_count, int) and isinstance(scope_eval_duration, int)
                             and scope_eval_duration > 0 else None)
        planner_eval_count = (planner_native.get("eval_count") if isinstance(planner_native, dict) else None)
        planner_eval_duration = (planner_native.get("eval_duration") if isinstance(planner_native, dict) else None)
        planner_decode_rate = (planner_eval_count * 1_000_000_000 / planner_eval_duration
                               if isinstance(planner_eval_count, int) and isinstance(planner_eval_duration, int)
                               and planner_eval_duration > 0 else None)
        predicted_search = (None if scope_error is not None or planner_error is not None else
                            scope_predicted == "schema" and planner_search is True)
        rows.append({
            "id": case["id"], "question": case["question"], "expected": case["expected"],
            "predicted_scope": scope_predicted, "scope_correct": scope_predicted == case["expected"],
            "scope_latency_ms": scope_latency,
            "scope_ollama_total_ms": native_ms(scope_native, "total_duration"),
            "scope_ollama_load_ms": native_ms(scope_native, "load_duration"),
            "scope_ollama_prompt_eval_ms": native_ms(scope_native, "prompt_eval_duration"),
            "scope_ollama_decode_ms": native_ms(scope_native, "eval_duration"),
            "scope_ollama_prompt_tokens": (scope_native.get("prompt_eval_count")
                                           if isinstance(scope_native, dict) else None),
            "scope_ollama_decode_tokens": scope_eval_count,
            "scope_ollama_decode_tokens_per_second": (round(scope_decode_rate, 2)
                                                       if scope_decode_rate is not None else None),
            "scope_error": scope_error,
            "expected_search": case["expected"] == "schema", "predicted_search": predicted_search,
            "planner_search": planner_search, "planner_query": planner_query,
            "query_changed": query_changed, "planner_latency_ms": planner_latency,
            "planner_ollama_total_ms": native_ms(planner_native, "total_duration"),
            "planner_ollama_load_ms": native_ms(planner_native, "load_duration"),
            "planner_ollama_prompt_eval_ms": native_ms(planner_native, "prompt_eval_duration"),
            "planner_ollama_decode_ms": native_ms(planner_native, "eval_duration"),
            "planner_ollama_prompt_tokens": (planner_native.get("prompt_eval_count")
                                              if isinstance(planner_native, dict) else None),
            "planner_ollama_decode_tokens": planner_eval_count,
            "planner_ollama_decode_tokens_per_second": (round(planner_decode_rate, 2)
                                                         if planner_decode_rate is not None else None),
            "planner_error": planner_error,
            "pipeline_latency_ms": (round(scope_latency + (planner_latency or 0), 2)
                                    if scope_error is None else None),
        })
        print(f"{name} {split} {position}/{len(cases)} {case['id']}: scope={scope_predicted or 'error'} "
              f"search={predicted_search} · scope {scope_latency} ms · planner {planner_latency} ms",
              flush=True)

    metrics = summarize(rows, list(SCOPE_CATEGORIES))
    result = {
        "model": name, "model_digest": model_digest_value, "model_size_bytes": model_info["size_bytes"],
        "parameter_size": model_info.get("parameter_size"), "quantization": model_info.get("quantization"),
        "split": split, "cold_start_scope_query_ms": cold_start_ms,
        "cold_start_scope_ollama_total_ms": native_ms(warmup_native, "total_duration"),
        "cold_start_scope_ollama_load_ms": native_ms(warmup_native, "load_duration"),
        "cold_start_scope_ollama_prompt_eval_ms": native_ms(warmup_native, "prompt_eval_duration"),
        "cold_start_scope_ollama_decode_ms": native_ms(warmup_native, "eval_duration"),
        "warmup_question": warmup_question, "warmup_prediction": warmup_prediction,
        "metrics": metrics, "cases": rows,
    }
    models.unload_if_loaded(name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/scope-questions-v1.json")
    parser.add_argument("--split", choices=("development", "test", "regression", "all"),
                        default="development")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Installed local Ollama chat models, in comparison order.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--warmup-question", default="How is a customer represented in the CDM schema?")
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        raise SystemExit("Choose each model only once.")

    dataset, raw, cases = load_cases(args.dataset, args.split)
    installed = {item["name"]: item for item in available_chat_models(args.base_url)}
    missing = [name for name in args.models if name not in installed]
    if missing:
        raise SystemExit("These models are not installed or exceed the configured benchmark size cap: "
                         + ", ".join(missing))

    unused = "__scope_classifier_benchmark_unused__"
    models = Ollama(args.base_url, os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
                    unused, unused, args.models[0], unused)
    model_results = []
    for name in args.models:
        model_results.append(run_model(models, args.base_url, installed[name], cases,
                                       args.split, args.warmup_question))

    report = {
        "run_id": hashlib.sha256((datetime.now(UTC).isoformat() + ":" + ",".join(args.models)
                                  + ":" + args.split).encode()).hexdigest()[:16],
        "created_at": datetime.now(UTC).isoformat(), "dataset_id": dataset.get("dataset_id"),
        "dataset_status": dataset.get("status"), "dataset_path": str(args.dataset), "split": args.split,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(), "case_count": len(cases),
        "scope_prompt_sha256": hashlib.sha256(SCOPE_PROMPT.encode()).hexdigest(),
        "scope_code_sha256": hashlib.sha256((ROOT / "app/scope.py").read_bytes()).hexdigest(),
        "router_code_sha256": hashlib.sha256((ROOT / "app/rag.py").read_bytes()).hexdigest(),
        "ollama_client_code_sha256": hashlib.sha256((ROOT / "app/ollama.py").read_bytes()).hexdigest(),
        "warmup_is_scored": False, "models": model_results,
        "notes": (
            "Each candidate is assigned to both the production scope classifier and local query planner. The "
            "scope warmup request is run after unloading each model and reported separately; per-question wall "
            "latency is measured after warmup. The planner runs only after a question is classified as schema; "
            "its one-call latency includes both the search decision and natural-language query creation. "
            "Ollama-native load, prompt-evaluation, decode durations, token counts, and decode rates are saved "
            "separately for both stages when available. Query text is retained for review but is not graded for "
            "retrieval quality. Dataset labels are assistant-authored and not independently reviewed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "split": args.split,
                      "models": {item["model"]: {
                          "cold_start_scope_query_ms": item["cold_start_scope_query_ms"],
                                                  **item["metrics"]} for item in model_results}},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
