#!/usr/bin/env python3
"""Compare answers from raw and rewritten queries on the same fixed cases.

This process calls Ollama directly and does not acquire a host model-work lease.
Run only in an admitted exclusive batch, with the serving app stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.benchmark import _judge_answer, _source_overlap
from app.retrieval.index import SchemaIndex
from app.runtime.ollama import Ollama
from app.retrieval.rag import cited_answer, format_entity_citations, prepare_passages, rewrite_search_query


def select_context(candidates: list[dict[str, Any]], limit: int,
                   strategy: str = "one_per_source") -> tuple[list, list]:
    """Apply the same deterministic passage selection policy to both arms."""
    if strategy not in {"one_per_source", "two_kinds_per_source"}:
        raise ValueError("Unknown context selection strategy")
    if limit < 1:
        raise ValueError("Context limit must be positive")
    selected: list[dict[str, Any]] = []
    kinds_by_source: dict[str, set[str]] = {}
    seen_ids: set[str] = set()
    for item in candidates:
        key = item.get("entity_id") or item.get("source_id") or item["id"]
        kind = item.get("kind", "")
        kinds = kinds_by_source.setdefault(key, set())
        if item["id"] in seen_ids or len(kinds) >= (1 if strategy == "one_per_source" else 2) or kind in kinds:
            continue
        seen_ids.add(item["id"])
        kinds.add(kind)
        selected.append(item)
        if len(selected) == limit:
            break
    contexts, sources, _ = prepare_passages(selected, limit)
    return contexts, sources


def run_case(case: dict[str, Any], index: Any, models: Any, judge_model: str,
             context_selection: str = "one_per_source") -> dict[str, Any]:
    """Keep failures explicit and independently evaluate each valid arm."""
    row: dict[str, Any] = {"id": case["id"], "question": case["question"],
                           "reference_answer": case["reference_answer"], "arms": {}}
    # Fixed before looking at labels, so the expected answer cannot affect retrieval.
    limit = 6
    history = case.get("context_messages", [])
    full_question = ("\n".join(f"{m['role'].title()}: {m['content']}" for m in history)
                     + f"\nUser: {case['question']}" if history else case["question"])
    rewrite = None
    rewrite_error = None
    started = time.perf_counter()
    try:
        rewrite = rewrite_search_query(case["question"], models, case.get("context_messages"))
    except Exception as exc:
        rewrite_error = f"{type(exc).__name__}: {exc}"
    rewrite_seconds = round(time.perf_counter() - started, 3)
    rewrite_runtime = models.runtime_info(models.router_model)
    models.unload_if_loaded(models.router_model)
    for arm, query in (("raw", full_question), ("rewrite", rewrite)):
        item: dict[str, Any] = {"query": query, "status": "pending", "error": None,
                                "rewrite_seconds": rewrite_seconds if arm == "rewrite" else 0.0,
                                "rewrite_runtime": rewrite_runtime if arm == "rewrite" else None}
        row["arms"][arm] = item
        if not query:
            item.update(status="error", error=rewrite_error or "empty rewrite", stage="rewrite")
            continue
        stage = "retrieval"
        try:
            started = time.perf_counter()
            candidates = index.search(query, limit=30, mode="hybrid")
            item["retrieval_seconds"] = round(time.perf_counter() - started, 3)
            item["candidates"] = candidates
            contexts, sources = select_context(candidates, limit, context_selection)
            item["contexts"] = contexts
            item["sources"] = sources
            if not sources:
                raise ValueError("no evidence retrieved")
            models.unload_if_loaded(models.embedding_model)
            stage = "answer"
            started = time.perf_counter()
            answer, generator = cited_answer(full_question, sources, contexts, models)
            item.update(raw_answer=answer, generator=generator,
                        answer_seconds=round(time.perf_counter() - started, 3),
                        answer_runtime=models.runtime_info(models.answer_model))
            if generator in {"none", "extractive-fallback", "error"}:
                raise ValueError(f"answer generation returned {generator}")
            overlap = _source_overlap(answer, sources, case["expected_entity_ids"])
            item.update(overlap)
            item["answer"], item["cited_sources"] = format_entity_citations(answer, sources)
            models.unload_if_loaded(models.answer_model)
            stage = "judge"
            started = time.perf_counter()
            item["judgment"] = _judge_answer(full_question, case["reference_answer"], item["answer"],
                                               case["expected_entity_ids"], overlap["cited_entity_ids"],
                                               sources, models, judge_model)
            item["judge_seconds"] = round(time.perf_counter() - started, 3)
            item["judge_runtime"] = models.runtime_info(judge_model)
            item["status"] = "scored"
        except Exception as exc:
            item.update(status="error", stage=stage, error=f"{type(exc).__name__}: {exc}")
        finally:
            for name in dict.fromkeys((models.embedding_model, models.answer_model, judge_model)):
                models.unload_if_loaded(name)
    return row


def paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [r for r in rows if all(r["arms"][a]["status"] == "scored" for a in ("raw", "rewrite"))]
    delta = [r["arms"]["rewrite"]["judgment"]["score"] - r["arms"]["raw"]["judgment"]["score"]
             for r in pairs]
    return {"total_cases": len(rows), "paired_scored": len(pairs),
            "improved": sum(x > 0 for x in delta), "tied": sum(x == 0 for x in delta),
            "regressed": sum(x < 0 for x in delta),
            "mean_judge_delta": round(sum(delta) / len(delta), 3) if delta else None}


def write_result(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def repository_metadata(supplied_commit: str | None) -> dict[str, Any]:
    """Report supplied provenance honestly when Git metadata is unavailable."""
    if supplied_commit:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", supplied_commit):
            raise ValueError("--repository-commit must be a full 40-character commit SHA")
        return {"repository_commit": supplied_commit.lower(), "repository_commit_source": "operator_supplied",
                "git_status_porcelain": None, "git_status": "unknown"}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                  capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain=v1"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"repository_commit": None, "repository_commit_source": "unavailable",
                "git_status_porcelain": None, "git_status": "unknown"}
    return {"repository_commit": revision, "repository_commit_source": "git",
            "git_status_porcelain": status, "git_status": "dirty" if status else "clean"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--embedding", default="nomic-embed-text:latest")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/answer-benchmark-v1.json")
    parser.add_argument("--catalog", type=Path, default=ROOT / "seed/catalog.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-commit", help="Full source commit SHA supplied by the operator when Git is unavailable")
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--context-selection", choices=("one_per_source", "two_kinds_per_source"),
                        default="one_per_source",
                        help="Passage selection for both arms; default reproduces the earlier experiment")
    args = parser.parse_args()
    payload = json.loads(args.dataset.read_text())
    catalog = json.loads(args.catalog.read_text())
    if payload.get("source_commit") != catalog.get("source_commit"):
        raise ValueError("Dataset and catalog source revisions differ")
    by_id = {q["id"]: q for q in payload["questions"]}
    ids = args.question_id or list(by_id)
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= by_id.keys():
        raise ValueError("Question IDs must be unique and present in the dataset")
    tags = httpx.get(args.base_url.rstrip("/") + "/api/tags", timeout=10, trust_env=False)
    tags.raise_for_status()
    available = {m["name"]: m for m in tags.json()["models"]}
    if any(name not in available for name in (args.embedding, args.router, args.answer, args.judge)):
        raise ValueError("Every selected model must be installed under its exact tag")
    repository = repository_metadata(args.repository_commit)
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write("{}\n")
    relevant_files = ["app/retrieval/index.py", "app/schema/corpus.py", "app/retrieval/rag.py", "app/runtime/ollama.py", "app/evaluation/benchmark.py"]
    result: dict[str, Any] = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        **repository, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in relevant_files},
        "dataset_id": payload.get("dataset_id"),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "source_commit": payload["source_commit"], "question_ids": ids,
        "host": {"system": platform.platform(), "machine": platform.machine()},
        "models": {role: available[name] for role, name in (("embedding", args.embedding),
                   ("router", args.router), ("answer", args.answer), ("judge", args.judge))},
        "method": "raw versus rewrite-only, hybrid top 30, up to six passages; same answer and judge",
        "context_selection": args.context_selection,
        "context_selection_description": (
            "at most one passage per entity or source" if args.context_selection == "one_per_source"
            else "at most two passages of distinct kinds per entity or source, in retrieval rank order"),
        "notes": "Direct Ollama calls require an externally admitted exclusive host slot; scores use incomplete references.",
        "rows": []}
    write_result(args.output, result)
    models = Ollama(args.base_url, args.embedding, args.router, args.answer, judge_model=args.judge)
    try:
        index = SchemaIndex(args.catalog, args.data_dir, models)
        result.update(index_revision=index.index_revision, index_reused=index.index_reused)
        models.unload_if_loaded(args.embedding)
        for qid in ids:
            print(f"Evaluating {qid}", flush=True)
            result["rows"].append(run_case(by_id[qid], index, models, args.judge, args.context_selection))
            result["summary"] = paired_summary(result["rows"])
            write_result(args.output, result)
        result["status"] = ("completed_with_errors" if any(
            arm["status"] != "scored" for row in result["rows"] for arm in row["arms"].values()
        ) else "completed")
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_result(args.output, result)


if __name__ == "__main__":
    main()
