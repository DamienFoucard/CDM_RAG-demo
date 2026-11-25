#!/usr/bin/env python3
"""Evaluate the preflight gate and local or Jev routing on saved probe scores."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ollama import Ollama
from app.rag import plan_routed_search
from app.router import JevRouter


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("ollama", "jev"),
                        default=os.getenv("CDM_RAG_ROUTER_BACKEND", "ollama"))
    parser.add_argument("--confidence-threshold", type=float,
                        default=float(os.getenv("CDM_RAG_ROUTER_MIN_CONFIDENCE", "0.70")))
    parser.add_argument("--input-cost-per-million", type=float,
                        help="Optional Jev input-token price in USD per million tokens.")
    args = parser.parse_args()
    if not 0 <= args.confidence_threshold <= 1:
        raise SystemExit("--confidence-threshold must be between 0 and 1")
    if (args.input_cost_per_million is not None
            and (not math.isfinite(args.input_cost_per_million) or args.input_cost_per_million < 0)):
        raise SystemExit("--input-cost-per-million must be a finite non-negative number")
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    models = Ollama(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"), benchmark["embedding_model"],
                    os.getenv("ROUTER_MODEL", "qwen3:4b"), os.getenv("ANSWER_MODEL", "qwen3:4b"))
    router = JevRouter(os.getenv("JEV_API_KEY", ""), os.getenv("JEV_MODEL", "jev-latest"),
                       os.getenv("JEV_BASE_URL", "https://api.typesafe.ai"), args.confidence_threshold) \
        if args.backend == "jev" else None
    threshold = benchmark["gate"]["threshold"]
    rows = []
    for case in benchmark["cases"] + benchmark["out_of_scope"]:
        expected = "positive_entity_ids" in case
        passed_probe = case["probe_similarity"] >= threshold
        if passed_probe:
            decision, query, routing = plan_routed_search(case["question"], models, router)
        else:
            decision, query = False, ""
            routing = {"router_backend": args.backend, "router_decision": "semantic_gate",
                       "router_latency_ms": 0.0, "router_classification_latency_ms": 0.0,
                       "query_generation_latency_ms": None, "router_fallback_used": False,
                       "query_generated": False}
        rows.append({"id": case["id"], "question": case["question"], "category": case["category"],
                     "in_scope": expected, "probe_passed": passed_probe, "search": decision,
                     "search_query": query,
                     "generated_query": query if routing["query_generated"] else "",
                     **routing})
        print(f'{len(rows)}: {case["id"]} search={decision}', flush=True)
    groups = {"in_scope": [x for x in rows if x["in_scope"]],
              "off_topic": [x for x in rows if not x["in_scope"] and x["category"] == "off_topic"],
              "unsupported_fact": [x for x in rows if not x["in_scope"] and x["category"] == "unsupported_fact"]}
    metrics = {name + "_search_rate": round(statistics.mean(x["search"] for x in cases), 4)
               for name, cases in groups.items() if cases}
    routed_rows = [row for row in rows if row["probe_passed"]]
    latencies = [float(row["router_latency_ms"]) for row in routed_rows]
    input_tokens = sum(int(row.get("router_usage_input_tokens", 0)) for row in routed_rows)
    metrics.update({
        "raw_search_rate": round(statistics.mean(row["router_decision"] == "raw_search"
                                                  for row in routed_rows), 4) if routed_rows else 0,
        "query_generation_rate": round(statistics.mean(bool(row["query_generated"])
                                                        for row in routed_rows), 4) if routed_rows else 0,
        "confidence_fallback_rate": round(statistics.mean(bool(row["router_fallback_used"])
                                                          for row in routed_rows), 4) if routed_rows else 0,
        "mean_router_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "p95_router_latency_ms": round(percentile(latencies, 0.95), 1) if latencies else None,
        "router_input_tokens": input_tokens if args.backend == "jev" else None,
        "estimated_jev_input_cost_usd": (
            round(input_tokens * args.input_cost_per_million / 1_000_000, 8)
            if args.backend == "jev" and args.input_cost_per_million is not None else None
        ),
    })
    result = {"dataset_id": benchmark["dataset_id"], "dataset_sha256": benchmark["dataset_sha256"],
              "dataset_status": benchmark["dataset_status"], "split": benchmark["split"],
              "embedding_model": benchmark["embedding_model"], "model_digest": benchmark["model_digest"],
              "router_backend": args.backend,
              "router_model": router.model if router is not None else models.router_model,
              "probe_threshold": threshold,
              "router_confidence_threshold": args.confidence_threshold if router is not None else None,
              "metrics": metrics,
              "notes": "Synthetic, unreviewed examples; this scores scope routing, route mix and latency, not query quality, retrieval or answers. Raw-search routes use the original wording. Query rewrites are natural-language retrieval queries, not SQL or SPARQL. Probe scores were measured separately.",
              "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if router is not None:
        router.close()
    print(json.dumps({"output": str(args.output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
