#!/usr/bin/env python3
"""Compare initial retrieval and LLM reranking on a provisional CDM question set."""
from __future__ import annotations

import argparse
import hashlib
import json
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
    rank = next((position for position, item in enumerate(ranked, 1) if item in gold), None)
    return {"ranked_entity_ids": ranked, "hit": rank is not None,
            "recall": len(gold.intersection(ranked)) / len(gold),
            "reciprocal_rank": 1 / rank if rank else 0.0}


def aggregate(cases: list[dict], variant: str) -> dict:
    rows = [case[variant] for case in cases]
    return {"cases": len(rows), "hit_at_6": round(statistics.mean(row["hit"] for row in rows), 4),
            "recall_at_6": round(statistics.mean(row["recall"] for row in rows), 4),
            "mrr_at_6": round(statistics.mean(row["reciprocal_rank"] for row in rows), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "cdm-schema-pilot.json")
    parser.add_argument("--cases", type=int, default=6, help="Number of questions; use 18 for the entire pilot.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    dataset = json.loads(raw)
    if not 1 <= args.cases <= len(dataset["questions"]):
        raise SystemExit("--cases must fit the question set")
    models = Ollama(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                    os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
                    os.getenv("ROUTER_MODEL", "qwen3:4b"), os.getenv("ANSWER_MODEL", "qwen3:4b"))
    index = SchemaIndex(ROOT / "seed" / "catalog.json", Path(os.getenv("CDM_RAG_DATA_DIR", ROOT / "data")), models)
    if dataset["source_commit"] != index.catalog["source_commit"]:
        raise SystemExit("the question set targets a different CDM source revision")
    cases = []
    for question in dataset["questions"][:args.cases]:
        if set(question["positive_entity_ids"]) - set(index.graph.entities):
            raise SystemExit("a labeled entity is missing from this catalog")
        started = time.perf_counter()
        candidates = index.search(question["question"], limit=20, mode="hybrid")
        initial = score(candidates[:6], question["positive_entity_ids"])
        ranked = rerank(question["question"], candidates, models, 6)
        after = score(ranked, question["positive_entity_ids"])
        cases.append({"id": question["id"], "question": question["question"],
                      "positive_entity_ids": question["positive_entity_ids"],
                      "initial": initial, "reranked": after,
                      "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})
        print(f"{len(cases)}/{args.cases}: {question['id']} initial={initial['hit']} reranked={after['hit']}", flush=True)
    result = {"dataset_id": dataset["dataset_id"], "dataset_status": dataset["status"],
              "dataset_provenance": dataset["provenance"], "dataset_sha256": hashlib.sha256(raw).hexdigest(),
              "source_commit": index.catalog["source_commit"], "index_revision": index.index_revision,
              "embedding_model": models.embedding_model, "reranker_model": models.answer_model,
              "notes": "The same raw question and candidates are used for both variants; routing, query rewriting, and answer generation are not measured. Unlisted entities are unjudged.",
              "metrics": {name: aggregate(cases, name) for name in ("initial", "reranked")}, "cases": cases}
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
