#!/usr/bin/env python3
"""Compare a router with saved question-only decisions after an upfront search.

The preliminary search is fixed hybrid BM25 plus Nomic embeddings; it does not
call an LLM or reranker. Each case records actual wall time for searching and
then classifying. The embedder and one small router may stay resident, but their
inference calls are sequential. The script does not acquire the app's model-work
lock and must be run in isolation.
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
from app.schema.corpus import load_catalog, render_chunks
from app.retrieval.decision_router import (EVIDENCE_HIT_LIMIT, EVIDENCE_SNIPPET_CHARS, ROUTES,
                                 ROUTE_PROMPT, ROUTE_SCHEMA, evidence_messages,
                                 initial_hits, initial_query)
from app.schema.graph import SchemaGraph
from app.retrieval.index import EMBEDDING_FRAGMENT_CHARS, SchemaIndex, _split_embedding_text
from app.runtime.ollama import ModelError
from app.evaluation.retrieval_benchmark import _slug
from scripts.benchmark_router_decisions import accepted_routes, save, summarize, validate_dataset
from scripts.comparison_ollama import ComparisonOllama, comparison_settings
from scripts.replay_reasoning_adaptive import load_completed

EMBEDDING = "nomic-embed-text:latest"
HIT_LIMIT = EVIDENCE_HIT_LIMIT
SNIPPET_CHARS = EVIDENCE_SNIPPET_CHARS
MAX_ROUTER_BYTES = 12_000_000_000


class WarmComparisonOllama(ComparisonOllama):
    """Keep the router warm just as the embedder is kept warm in this run."""

    def _chat_payload(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = super()._chat_payload(*args, **kwargs)
        payload["keep_alive"] = "5m"
        return payload


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_baseline(dataset: dict[str, Any], raw: bytes, baseline: dict[str, Any],
                      tags: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    if (baseline.get("status") != "completed" or baseline.get("split") != "test"
            or baseline.get("dataset_id") != dataset["dataset_id"]
            or baseline.get("dataset_sha256") != sha(raw)
            or baseline.get("prompt_sha256") != sha(ROUTE_PROMPT.encode())):
        raise ValueError("The saved question-only baseline does not match the dataset or prompt.")
    settings = baseline.get("model_settings", {})
    if settings.get("think") is not False or settings.get("max_output_tokens") != 160:
        raise ValueError("The saved question-only baseline uses different inference settings.")
    cases = {c["id"]: c for c in validate_dataset(dataset) if c["split"] == "test"}
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in baseline.get("rows", []):
        key = (row.get("model"), row.get("case_id"))
        if key[0] not in tags:
            continue
        case = cases.get(key[1])
        if (not case or key in output or row.get("question") != case["question"]
                or row.get("context_messages", []) != case.get("context_messages", [])
                or accepted_routes(row) != accepted_routes(case)):
            raise ValueError("A saved baseline row differs from its dataset.")
        output[key] = row
    if set(output) != {(tag, cid) for tag in tags for cid in cases}:
        raise ValueError("The saved baseline does not cover all requested models and test cases.")
    return output


def cached_index_preflight(catalog_path: Path, index_dir: Path, models: Any,
                           expected_identity: str, expected_revision: str) -> None:
    """Refuse a mismatch before SchemaIndex could create or re-embed a collection."""
    catalog = load_catalog(catalog_path)
    if catalog.get("source_commit") != expected_revision:
        raise ValueError("The pinned catalog source revision differs.")
    SchemaGraph(catalog["entities"], catalog["source_commit"], semantics=catalog.get("semantics", {}))
    chunks = render_chunks(catalog)
    payload = {"chunks": chunks, "model": models.embedding_model,
               "model_digest": models.embedding_digest(), "format": "cdm-passages-v3",
               "embedding_fragment_chars": EMBEDDING_FRAGMENT_CHARS,
               "embedding_format": models.embedding_format()}
    identity = sha(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())
    if identity != expected_identity:
        raise ValueError("The cached index identity differs from the saved retrieval benchmark.")
    vector_ids = set()
    for chunk in chunks:
        text = chunk["text"]
        if len(text) <= EMBEDDING_FRAGMENT_CHARS:
            vector_ids.add(chunk["id"])
        else:
            prefix = f"{chunk.get('title') or chunk['id']}: "[:EMBEDDING_FRAGMENT_CHARS // 4]
            parts = _split_embedding_text(text, EMBEDDING_FRAGMENT_CHARS - len(prefix))
            vector_ids.update(f"{chunk['id']}::fragment:{n}" for n in range(1, len(parts) + 1))
    if not (index_dir / "chroma").exists():
        raise ValueError("The verified cached retrieval index is unavailable.")
    import chromadb
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    try:
        collection = client.get_collection("cdm_" + identity[:24], embedding_function=None)
        if set(collection.get(include=[]).get("ids", [])) != vector_ids:
            raise ValueError("The cached index is incomplete; this experiment will not rebuild it.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The expected cached index collection is unavailable.") from exc


def run_case(case: dict[str, Any], model: Any, index: Any, baseline: dict[str, Any]) -> dict[str, Any]:
    """Run one actual search-to-classification pipeline without concurrent inference."""
    total_start = time.perf_counter()
    search_seconds = 0.0
    router_seconds = 0.0
    query = initial_query(case)
    row: dict[str, Any] = {
        "model": model.router_model, "case_id": case["id"], "question": case["question"],
        "context_messages": case.get("context_messages", []), "expected_route": case.get("expected_route"),
        "accepted_routes": accepted_routes(case), "initial_query": query, "initial_hits": [],
        "baseline_route": baseline.get("predicted_route"), "baseline_status": baseline.get("status"),
        "baseline_correct": baseline.get("correct"), "baseline_seconds": baseline.get("elapsed_seconds"),
    }
    try:
        search_start = time.perf_counter()
        row["initial_hits"] = initial_hits(index.search(query, limit=HIT_LIMIT, mode="hybrid"))
        search_seconds = time.perf_counter() - search_start
        row["search_runtime"] = model.runtime_info(model.embedding_model)
        messages = evidence_messages(case, row["initial_hits"])
        row["messages_sha256"] = sha(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode())
        router_start = time.perf_counter()
        result, timing = model.chat_json_timed(model.router_model, messages, schema=ROUTE_SCHEMA, max_tokens=160)
        router_seconds = time.perf_counter() - router_start
        if not isinstance(result, dict) or set(result) != {"route"} or result["route"] not in ROUTES:
            raise ModelError("The local model returned an invalid routing decision.")
        prediction = result["route"]
        row.update(status="scored", predicted_route=prediction, correct=prediction in row["accepted_routes"],
                   native_timing=timing, runtime=model.runtime_info(model.router_model))
    except (ModelError, ValueError, RuntimeError) as exc:
        if 'search_start' in locals() and 'router_start' not in locals() and not search_seconds:
            search_seconds = time.perf_counter() - search_start
        if 'router_start' in locals() and not router_seconds:
            router_seconds = time.perf_counter() - router_start
        row.update(status="error", predicted_route=None, correct=False, error=f"{type(exc).__name__}: {exc}")
    row.update(search_seconds=round(search_seconds, 3), router_seconds=round(router_seconds, 3),
               end_to_end_seconds=round(time.perf_counter() - total_start, 3))
    row["elapsed_seconds"] = row["end_to_end_seconds"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path, help="Completed saved question-only result")
    parser.add_argument("--index-baseline", required=True, type=Path, help="Completed saved adaptive SSE index run")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory with copied retrieval_benchmark_indices")
    parser.add_argument("--model", required=True, action="append", help="Installed router tag; repeat for each model")
    parser.add_argument("--case-id", action="append", help="Optional subset of test cases")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.model) != len(set(args.model)):
        parser.error("Model tags must be distinct")
    raw = args.dataset.read_bytes()
    dataset = json.loads(raw)
    all_cases = [case for case in validate_dataset(dataset) if case["split"] == "test"]
    cases = all_cases
    if args.case_id:
        if len(args.case_id) != len(set(args.case_id)) or not set(args.case_id) <= {c['id'] for c in cases}:
            parser.error("Case IDs must be distinct and belong to the test split")
        cases = [c for c in cases if c["id"] in args.case_id]
    if not cases:
        parser.error("No cases selected")
    baseline_raw = args.baseline.read_bytes()
    baseline = json.loads(baseline_raw)
    baseline_rows = validate_baseline(dataset, raw, baseline, args.model)
    response = httpx.get(args.base_url.rstrip("/") + "/api/tags", timeout=10, trust_env=False)
    response.raise_for_status()
    available = {item["name"]: item for item in response.json()["models"]}
    if EMBEDDING not in available or any(tag not in available for tag in args.model):
        raise ValueError("All exact model tags and the Nomic embedding model must be installed.")
    for tag in args.model:
        size = available[tag].get("size")
        if not isinstance(size, int) or size <= 0 or size > MAX_ROUTER_BYTES:
            raise ValueError("The warm co-resident experiment requires router weights at most 12 GB.")
        if baseline.get("models", {}).get(tag, {}).get("digest") != available[tag].get("digest"):
            raise ValueError("A router model digest differs from its question-only baseline.")
    index_run = load_completed(args.index_baseline)
    expected_identity = index_run["summaries"][0]["index_identity"]
    expected_revision = index_run["source_commit"]
    if dataset.get("source_commit") not in (None, expected_revision):
        raise ValueError("Dataset and cached catalog source revisions differ.")
    index_dir = args.data_dir / "retrieval_benchmark_indices" / _slug(EMBEDDING)
    model = WarmComparisonOllama(args.base_url, EMBEDDING, args.model[0], "__unused_answer__",
                                 judge_model="__unused_judge__")
    model.query_embedding_keep_alive = "5m"
    cached_index_preflight(ROOT / "seed/catalog.json", index_dir, model, expected_identity, expected_revision)
    index = SchemaIndex(ROOT / "seed/catalog.json", index_dir, model)
    if not index.index_reused or index.index_revision != expected_identity:
        raise ValueError("The verified cache changed before the experiment began.")
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    result: dict[str, Any] = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "method": "Unconditionally search raw conversation and question using cached Nomic hybrid index; present top three bounded excerpts to an independent classifier. No LLM reranker, rewrite or answer generation.",
        "caveat": "All cases are searched before classification, including out-of-scope requests. False-search measures the classifier's decision, not avoided initial searches. The embedder and one small router stay resident with 5-minute keep-alive, but do not infer concurrently. Timings include actual first-case model loads and other residency effects. Data are provisional and not independently reviewed.",
        "dataset_id": dataset["dataset_id"], "dataset_sha256": sha(raw), "split": "test",
        "selected_case_ids": [c["id"] for c in cases], "baseline_sha256": sha(baseline_raw),
        "baseline": {"summaries": {tag: summarize([baseline_rows[(tag, case["id"])] for case in cases])
                                   for tag in args.model},
                     "classifier_sha256": baseline.get("classifier_sha256"), "prompt_sha256": baseline["prompt_sha256"]},
        "index_identity": expected_identity, "source_commit": expected_revision,
        "index_baseline_sha256": sha(args.index_baseline.read_bytes()), "embedding": available[EMBEDDING],
        "models": {tag: available[tag] for tag in args.model},
        "search": {"mode": "hybrid", "reranker": None, "top_k": HIT_LIMIT, "snippet_chars": SNIPPET_CHARS,
                   "query_method": "raw prior conversation plus current question", "concurrent_inference": False,
                   "classifier_hit_fields": ["rank", "id", "title", "text"],
                   "embedder_keep_alive": "5m", "router_keep_alive": "5m",
                   "max_router_weight_bytes": MAX_ROUTER_BYTES},
        "model_settings": comparison_settings(None, None, 160, None),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "classifier_sha256": sha((ROOT / "app/retrieval/decision_router.py").read_bytes()),
        "helper_sha256": sha((ROOT / "scripts/comparison_ollama.py").read_bytes()),
        "host": {"system": platform.platform(), "machine": platform.machine()}, "rows": [], "summaries": {},
    }
    save(args.output, result)
    try:
        for tag in args.model:
            model.router_model = tag
            model_rows: list[dict[str, Any]] = []
            try:
                for case in cases:
                    print(f"{tag}: {case['id']}", flush=True)
                    row = run_case(case, model, index, baseline_rows[(tag, case["id"])])
                    model_rows.append(row)
                    result["rows"].append(row)
                    summary = summarize(model_rows)
                    summary.update(mean_end_to_end_seconds=summary["mean_seconds"],
                                   mean_search_seconds=sum(r["search_seconds"] for r in model_rows) / len(model_rows),
                                   mean_router_seconds=sum(r["router_seconds"] for r in model_rows) / len(model_rows))
                    result["summaries"][tag] = summary
                    save(args.output, result)
            finally:
                model.unload_if_loaded(tag)
        result["status"] = "completed_with_errors" if any(r["status"] != "scored" for r in result["rows"]) else "completed"
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            model.unload_if_loaded(EMBEDDING)
        except ModelError as exc:
            result["embedding_unload_error"] = str(exc)
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        save(args.output, result)


if __name__ == "__main__":
    main()
