#!/usr/bin/env python3
"""Compare retrieval configurations on a pinned, explicitly provisional corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval.index import SchemaIndex
from app.runtime.ollama import Ollama
from app.retrieval.rag import rerank

ROOT = Path(__file__).resolve().parents[1]


def score(candidates: list[dict], positives: list[str]) -> dict:
    ranked = list(dict.fromkeys(item["entity_id"] for item in candidates))
    gold = set(positives)
    if not gold:
        raise ValueError("a retrieval case must have at least one judged positive")
    first = next((i for i, item in enumerate(ranked, 1) if item in gold), None)
    return {"ranked_entity_ids": ranked, "hit_at_1": bool(first and first <= 1),
            "hit_at_3": bool(first and first <= 3), "hit_at_6": first is not None,
            "recall_at_6": len(gold.intersection(ranked)) / len(gold),
            "mrr_at_6": 1 / first if first else 0.0}


def aggregate(cases: list[dict], variant: str) -> dict:
    rows = [case["variants"][variant] for case in cases]
    latencies = sorted(row["latency_ms"] for row in rows)
    return {"cases": len(rows), **{key: round(statistics.mean(row[key] for row in rows), 4)
            for key in ("hit_at_1", "hit_at_3", "hit_at_6", "recall_at_6", "mrr_at_6")},
            "mean_latency_ms": round(statistics.mean(latencies), 2),
            "p95_latency_ms": round(latencies[math.ceil(0.95 * len(latencies)) - 1], 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/cdm-schema-evaluation-v2.json")
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--model", required=True, help="An installed, explicitly supported Ollama embedding model.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Isolated benchmark index directory.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.60, help="Semantic probe threshold; do not tune on test.")
    parser.add_argument("--rerank", action="store_true", help="Also evaluate the LLM reranker; this is slow.")
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    dataset = json.loads(raw)
    questions = [q for q in dataset["questions"] if q.get("split") == args.split]
    negatives = [q for q in dataset["out_of_scope"] if q.get("split") == args.split]
    if not questions or not negatives or not -1 <= args.threshold <= 1:
        raise SystemExit("A valid split, out-of-scope cases, and threshold are required.")
    models = Ollama(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"), args.model,
                    os.getenv("ROUTER_MODEL", "qwen3:4b"), os.getenv("ANSWER_MODEL", "qwen3:4b"))
    print(f"Preparing isolated index for {args.model}", flush=True)
    index_started = time.perf_counter()
    index = SchemaIndex(ROOT / "seed/catalog.json", args.data_dir, models)
    index_seconds = round(time.perf_counter() - index_started, 2)
    if dataset["source_commit"] != index.catalog["source_commit"]:
        raise SystemExit("the question set targets a different CDM source revision")
    if any(set(q["positive_entity_ids"]) - set(index.graph.entities) for q in questions):
        raise SystemExit("a labeled entity is missing from this catalog")
    variants = ["lexical", "semantic", "hybrid"] + (["hybrid_reranked"] if args.rerank else [])
    # Warm the embedding and query path; timings below are warm, local searches.
    index.search(questions[0]["question"], limit=20, mode="hybrid")
    cases = []
    for question in questions:
        measured = {}
        for mode in ("lexical", "semantic", "hybrid"):
            start = time.perf_counter()
            candidates = index.search(question["question"], limit=20, mode=mode)
            elapsed = (time.perf_counter() - start) * 1000
            measured[mode] = {**score(candidates[:6], question["positive_entity_ids"]),
                              "latency_ms": round(elapsed, 2)}
            if mode == "hybrid" and args.rerank:
                start = time.perf_counter()
                selected = rerank(question["question"], candidates, models, 6)
                rerank_ms = (time.perf_counter() - start) * 1000
                measured["hybrid_reranked"] = {**score(selected, question["positive_entity_ids"]),
                                                "latency_ms": round(elapsed + rerank_ms, 2)}
        similarity = index.semantic_probe(question["question"])
        cases.append({**question, "probe_similarity": round(similarity, 6), "variants": measured})
        print(f"{args.split}: {len(cases)}/{len(questions)} {question['id']}", flush=True)
    negative_results = [{**item, "probe_similarity": round(index.semantic_probe(item["question"]), 6)}
                        for item in negatives]
    group = lambda category: [item for item in negative_results if item["category"] == category]
    gate = {"threshold": args.threshold,
            "in_scope_pass_rate": round(statistics.mean(c["probe_similarity"] >= args.threshold for c in cases), 4)}
    for category in ("off_topic", "unsupported_fact"):
        items = group(category)
        gate[category + "_pass_rate"] = round(statistics.mean(c["probe_similarity"] >= args.threshold for c in items), 4)
    result = {"dataset_id": dataset["dataset_id"], "dataset_status": dataset["status"],
              "dataset_provenance": dataset["provenance"], "dataset_sha256": hashlib.sha256(raw).hexdigest(),
              "source_commit": index.catalog["source_commit"], "index_revision": index.index_revision,
              "model_digest": index.model_digest, "embedding_model": args.model,
              "embedding_format": models.embedding_format(), "split": args.split,
              "reranker_model": models.answer_model if args.rerank else None,
              "chunk_count": len(index.chunks), "candidate_limit": 20, "top_k_chunks": 6,
              "index_preparation_seconds": index_seconds, "gate": gate,
              "notes": "Labels are incomplete and unreviewed; other entities are unjudged. Ranking metrics deduplicate the first six chunks by entity. Timings are warm local search calls, not API or end-to-end answer latency. The probe is only the preliminary similarity gate, not the LLM router. Index preparation may reuse a prior index.",
              "metrics": {name: aggregate(cases, name) for name in variants},
              "cases": cases, "out_of_scope": negative_results}
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": result["metrics"], "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
