#!/usr/bin/env python3
"""Benchmark a development-tuned cosine-only search gate, without an LLM.

This local experiment uses the pinned Nomic index and exactly one embedding
probe per case. It searches when the nearest-passage cosine similarity is at
least the threshold; otherwise it returns no_search. It never rewrites.
Threshold selection uses only development questions from the three-way router
dataset and the linked scope source, then both test sets are scored unchanged.
The script does not acquire the app's model-work lock and must run in isolation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.decision_router import initial_query
from app.index import SchemaIndex
from app.ollama import ModelError, Ollama
from app.retrieval_benchmark import _slug
from scripts.benchmark_evidence_router import cached_index_preflight
from scripts.benchmark_router_decisions import accepted_routes, save, summarize, validate_dataset
from scripts.replay_reasoning_adaptive import load_completed

EMBEDDING = "nomic-embed-text:latest"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def expected_search(case: dict[str, Any]) -> bool:
    """Map valid three-way or binary labels to the catalog-search gate."""
    return accepted_routes(case) != ["no_search"]


def scope_development_cases(scope: dict[str, Any], source: dict[str, Any], raw: bytes) -> list[dict[str, Any]]:
    """Recover binary development labels only from the binary dataset's pinned source."""
    if (scope.get("source_dataset") != source.get("dataset_id")
            or scope.get("source_sha256") != sha(raw)):
        raise ValueError("The scope dataset does not match its original source and checksum.")
    original = source.get("cases")
    if not isinstance(original, list):
        raise ValueError("The scope source contains no cases.")
    by_id = {case.get("id"): case for case in original if isinstance(case, dict)}
    if len(by_id) != len(original):
        raise ValueError("The scope source contains duplicate or invalid IDs.")
    for case in validate_dataset(scope):
        old = by_id.get(case.get("source_id") or case["id"])
        if (not old or old.get("split") != case["split"] or old.get("question") != case["question"]
                or not isinstance(old.get("allow_schema_search"), bool)
                or expected_search(case) != old["allow_schema_search"]):
            raise ValueError("A derived scope case does not match its source label.")
    development = []
    for old in original:
        if old.get("split") != "development":
            continue
        allowed = old.get("allow_schema_search")
        if not isinstance(allowed, bool) or not isinstance(old.get("question"), str):
            raise ValueError("Invalid source development question or label.")
        development.append({"id": old["id"], "split": "development", "question": old["question"],
                            "context_messages": old.get("context_messages", []),
                            "accepted_routes": ["raw_search", "rewrite_search"] if allowed else ["no_search"]})
    if not development:
        raise ValueError("The source scope dataset has no development examples.")
    return development


def choose_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Maximize balanced gate accuracy; resolve ties toward the highest threshold."""
    if not rows or any(row.get("status") != "scored" for row in rows):
        raise ValueError("Every development probe must succeed before threshold selection.")
    if any(not isinstance(row.get("similarity"), (int, float)) or isinstance(row.get("similarity"), bool)
           or not math.isfinite(row["similarity"]) or not -1 <= row["similarity"] <= 1 for row in rows):
        raise ValueError("Development similarities must be finite cosine values.")
    if any(not isinstance(row.get("expected_search"), bool) for row in rows):
        raise ValueError("Development search labels must be boolean.")
    positives = [row for row in rows if row["expected_search"]]
    negatives = [row for row in rows if not row["expected_search"]]
    if not positives or not negatives:
        raise ValueError("Threshold calibration needs both search and no-search examples.")
    candidates = sorted({float(row["similarity"]) for row in rows})
    candidates.append(math.nextafter(candidates[-1], math.inf))  # also evaluate declining every case
    best: tuple[Fraction, float, Fraction, Fraction] | None = None
    for threshold in candidates:
        search_recall = Fraction(sum(row["similarity"] >= threshold for row in positives), len(positives))
        no_search_recall = Fraction(sum(row["similarity"] < threshold for row in negatives), len(negatives))
        balanced = (search_recall + no_search_recall) / 2
        candidate = (balanced, threshold, search_recall, no_search_recall)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return {"threshold": best[1], "balanced_accuracy": float(best[0]), "search_recall": float(best[2]),
            "no_search_recall": float(best[3]), "development_cases": len(rows),
            "search_cases": len(positives), "no_search_cases": len(negatives),
            "rule": "Search if nearest-passage cosine >= threshold. Maximize mean search and no-search recall on development; on ties choose the highest threshold."}


def cosine_probe_with_passage(index: Any, query: str) -> dict[str, Any]:
    """Use the semantic-probe calculation once, retaining its nearest fragment identity."""
    vector = index.models.embed([query], query=True)[0]
    result = index.collection.query(query_embeddings=[vector], n_results=1, include=["distances"])
    distances, ids = result.get("distances") or [], result.get("ids") or []
    if not distances or not distances[0] or not ids or not ids[0]:
        raise ModelError("The nearest-passage cosine probe returned no result.")
    fragment_id = ids[0][0]
    chunk_id = index.vector_to_chunk.get(fragment_id)
    passage = index.by_id.get(chunk_id)
    fragments = [record for record in index.embedding_records if record["id"] == fragment_id]
    if not isinstance(fragment_id, str) or not chunk_id or not passage or len(fragments) != 1:
        raise ValueError("The nearest vector could not be matched to the pinned passage and fragment.")
    distance = float(distances[0][0])
    if not math.isfinite(distance):
        raise ValueError("The nearest-passage cosine distance is not finite.")
    similarity = max(-1.0, min(1.0, 1.0 - distance))
    source = {key: passage[key] for key in ("id", "title", "uri", "source_id", "source_label", "kind")
              if isinstance(passage.get(key), str)}
    return {"similarity": similarity, "nearest_fragment_id": fragment_id, "nearest_passage_id": chunk_id,
            "nearest_fragment_text": fragments[0]["text"], "nearest_passage": source}


def probe_case(case: dict[str, Any], source_id: str, index: Any, model: Any) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {"case_id": case["id"], "source_dataset": source_id, "split": case["split"],
                           "question": case["question"], "context_messages": case.get("context_messages", []),
                           "expected_route": case.get("expected_route"), "accepted_routes": accepted_routes(case),
                           "expected_search": expected_search(case), "query": initial_query(case)}
    try:
        probe = cosine_probe_with_passage(index, row["query"])
        similarity = probe["similarity"]
        if (not isinstance(similarity, (int, float)) or isinstance(similarity, bool)
                or not math.isfinite(similarity) or not -1 <= similarity <= 1):
            raise ValueError("The semantic probe did not return a finite cosine similarity.")
        row.update(status="scored", **probe, runtime=model.runtime_info(EMBEDDING))
    except (ModelError, ValueError, RuntimeError) as exc:
        row.update(status="error", similarity=None, error=f"{type(exc).__name__}: {exc}")
    row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return row


def score_test(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    for row in rows:
        route = ("raw_search" if row["similarity"] >= threshold else "no_search") if row["status"] == "scored" else None
        row["predicted_route"] = route
        row["correct"] = route in row["accepted_routes"] if route is not None else False
    return summarize(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-dataset", type=Path, default=ROOT / "evaluation/routing-decision-v1.json")
    parser.add_argument("--scope-dataset", type=Path, default=ROOT / "evaluation/routing-scope-binary-v1.json")
    parser.add_argument("--scope-source", type=Path, default=ROOT / "evaluation/scope-questions-v1.json")
    parser.add_argument("--index-baseline", required=True, type=Path, help="Saved adaptive run identifying the pinned index")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing the verified retrieval_benchmark_indices")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    paths = {"routing": args.routing_dataset, "scope": args.scope_dataset, "scope_source": args.scope_source}
    raw = {key: path.read_bytes() for key, path in paths.items()}
    routing, scope, scope_source = (json.loads(raw[key]) for key in ("routing", "scope", "scope_source"))
    routing_cases, scope_cases = validate_dataset(routing), validate_dataset(scope)
    scope_dev = scope_development_cases(scope, scope_source, raw["scope_source"])
    development = [(routing["dataset_id"], case) for case in routing_cases if case["split"] == "development"]
    development += [(scope_source["dataset_id"], case) for case in scope_dev]
    tests = {routing["dataset_id"]: [c for c in routing_cases if c["split"] == "test"],
             scope["dataset_id"]: [c for c in scope_cases if c["split"] == "test"]}
    if not development or any(not values for values in tests.values()):
        raise ValueError("Both development and each test split must have cases.")
    development_queries = {" ".join(initial_query(case).casefold().split()) for _, case in development}
    test_queries = {" ".join(initial_query(case).casefold().split()) for cases in tests.values() for case in cases}
    if development_queries & test_queries:
        raise ValueError("An identical raw conversation appears in both development and test splits.")
    baseline_raw = args.index_baseline.read_bytes()
    baseline = load_completed(args.index_baseline)
    expected_identity = baseline["summaries"][0]["index_identity"]
    expected_revision = baseline["source_commit"]
    if routing.get("source_commit") not in (None, expected_revision):
        raise ValueError("The routing dataset and index use different source revisions.")
    response = httpx.get(args.base_url.rstrip("/") + "/api/tags", timeout=10, trust_env=False)
    response.raise_for_status()
    models = {m["name"]: m for m in response.json()["models"]}
    if EMBEDDING not in models:
        raise ValueError("The exact Nomic embedding tag must be installed.")
    model = Ollama(args.base_url, EMBEDDING, "__unused_router__", "__unused_answer__")
    model.query_embedding_keep_alive = "5m"
    index_dir = args.data_dir / "retrieval_benchmark_indices" / _slug(EMBEDDING)
    cached_index_preflight(ROOT / "seed/catalog.json", index_dir, model, expected_identity, expected_revision)
    index = SchemaIndex(ROOT / "seed/catalog.json", index_dir, model)
    if not index.index_reused or index.index_revision != expected_identity:
        raise ValueError("The verified cached index changed before the experiment began.")
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    result: dict[str, Any] = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "method": "One raw-conversation embedding and nearest-fragment cosine gate; search as written or no_search. This is not the hybrid top result. No LLM, rewrite or reranker.",
        "caveat": "All cases incur the preliminary embedding/probe, including out-of-scope cases. One threshold is selected only on the combined development examples. Similarity is not proof a request is safe or answerable.",
        "script_sha256": sha(Path(__file__).read_bytes()), "dataset_sha256": {key: sha(value) for key, value in raw.items()},
        "source_commit": expected_revision, "index_identity": expected_identity, "index_baseline_sha256": sha(baseline_raw),
        "embedding": models[EMBEDDING], "calibration_sources": [routing["dataset_id"], scope_source["dataset_id"]],
        "development": {"rows": []}, "datasets": {},
    }
    save(args.output, result)
    try:
        for source_id, case in development:
            print(f"development: {source_id} / {case['id']}", flush=True)
            result["development"]["rows"].append(probe_case(case, source_id, index, model))
            save(args.output, result)
        calibration = choose_threshold(result["development"]["rows"])
        result["calibration"] = calibration
        save(args.output, result)
        for dataset_id, cases in tests.items():
            group: dict[str, Any] = {"split": "test", "rows": []}
            result["datasets"][dataset_id] = group
            for case in cases:
                print(f"test: {dataset_id} / {case['id']}", flush=True)
                group["rows"].append(probe_case(case, dataset_id, index, model))
                save(args.output, result)
            group["summary"] = score_test(group["rows"], calibration["threshold"])
            save(args.output, result)
        result["status"] = ("completed_with_errors" if any(row["status"] != "scored"
                            for group in result["datasets"].values() for row in group["rows"]) else "completed")
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            model.unload_if_loaded(EMBEDDING)
        finally:
            result["completed_at"] = datetime.now(timezone.utc).isoformat()
            save(args.output, result)


if __name__ == "__main__":
    main()
