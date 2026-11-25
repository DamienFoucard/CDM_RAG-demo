"""Retrieval-only benchmark for comparing local CDM embedding models."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from app.evaluation.benchmark import max_model_size_bytes
from app.retrieval.index import SchemaIndex
from app.runtime.ollama import ModelError, Ollama


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "evaluation" / "cdm-schema-evaluation-v2.json"
ANSWER_DATASET_PATH = ROOT / "evaluation" / "answer-benchmark-v1.json"
OBJECT_DATASET_PATH = ROOT / "evaluation" / "catalog-object-retrieval-v1.json"
CONTEXTUAL_DATASET_PATH = ROOT / "evaluation" / "contextual-retrieval-v1.json"
DATA_DIR = Path(os.getenv("CDM_RAG_DATA_DIR", str(ROOT / "data")))
RETRIEVAL_BENCHMARK_LOG_PATH = DATA_DIR / "retrieval_benchmark_runs.jsonl"
_LOG_LOCK = threading.Lock()
MAX_SAVED_RUNS_IN_PAGE = 50
SUPPORTED_EMBEDDING_FAMILIES = {"nomic-embed-text", "qwen3-embedding"}


def load_retrieval_dataset() -> dict[str, Any]:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if (not isinstance(payload, dict) or not isinstance(questions, list) or not questions
            or any(not isinstance(item, dict) or not isinstance(item.get("id"), str)
                   or not isinstance(item.get("question"), str)
                   or item.get("split") not in {"development", "test"}
                   or not isinstance(item.get("positive_entity_ids"), list)
                   or not item["positive_entity_ids"]
                   or any(not isinstance(entity_id, str) for entity_id in item["positive_entity_ids"])
                   for item in questions)):
        raise ValueError("The local retrieval benchmark dataset is invalid.")
    if len({item["id"] for item in questions}) != len(questions):
        raise ValueError("The local retrieval benchmark contains duplicate question IDs.")
    answer_dataset = json.loads(ANSWER_DATASET_PATH.read_text(encoding="utf-8"))
    if (not isinstance(answer_dataset, dict)
            or answer_dataset.get("source_commit") != payload.get("source_commit")
            or not isinstance(answer_dataset.get("questions"), list)):
        raise ValueError("The supplemental retrieval questions do not match the pinned CDM revision.")
    complex_questions = []
    for item in answer_dataset["questions"]:
        if not isinstance(item, dict) or item.get("complexity") != "three-entity-comparison":
            continue
        entity_ids = item.get("expected_entity_ids")
        if (not isinstance(item.get("id"), str) or not isinstance(item.get("question"), str)
                or not isinstance(entity_ids, list) or len(entity_ids) < 2
                or any(not isinstance(entity_id, str) for entity_id in entity_ids)):
            raise ValueError("A supplemental complex retrieval question is invalid.")
        complex_questions.append({"id": item["id"], "question": item["question"], "split": "complex",
                                 "category": item["complexity"], "positive_entity_ids": entity_ids})
    if {item["id"] for item in questions}.intersection(item["id"] for item in complex_questions):
        raise ValueError("The retrieval benchmark contains duplicate supplemental question IDs.")
    payload["base_dataset_id"] = payload["dataset_id"]
    payload["supplemental_dataset_id"] = answer_dataset["dataset_id"]
    payload["supplemental_provenance"] = answer_dataset.get("provenance")
    payload["dataset_id"] = f"{payload['dataset_id']}+{answer_dataset['dataset_id']}-complex"
    payload["questions"] = questions + complex_questions
    return payload


def load_catalog_object_dataset() -> dict[str, Any]:
    """Load source-labeled cases for indexed CDM objects beyond entity summaries."""
    payload = json.loads(OBJECT_DATASET_PATH.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if (not isinstance(payload, dict) or not isinstance(payload.get("dataset_id"), str)
            or not isinstance(payload.get("source_commit"), str)
            or not isinstance(questions, list) or not questions
            or any(not isinstance(item, dict) or not isinstance(item.get("id"), str)
                   or not isinstance(item.get("question"), str)
                   or item.get("split") not in {"development", "test"}
                   or not isinstance(item.get("object_type"), str)
                   or not isinstance(item.get("expected_source_type"), str)
                   or not isinstance(item.get("expected_kind"), str)
                   or not isinstance(item.get("positive_passage_ids"), list)
                   or not item["positive_passage_ids"]
                   or any(not isinstance(passage_id, str) for passage_id in item["positive_passage_ids"])
                   for item in questions)):
        raise ValueError("The local catalog-object retrieval benchmark dataset is invalid.")
    if len({item["id"] for item in questions}) != len(questions):
        raise ValueError("The catalog-object retrieval benchmark contains duplicate question IDs.")
    if {item["split"] for item in questions} != {"development", "test"}:
        raise ValueError("The catalog-object retrieval benchmark must include development and test cases.")
    return payload


def load_contextual_retrieval_dataset() -> dict[str, Any]:
    """Load follow-up queries with prior messages and source-checked labels."""
    payload = json.loads(CONTEXTUAL_DATASET_PATH.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if (not isinstance(payload, dict) or not isinstance(payload.get("dataset_id"), str)
            or not isinstance(payload.get("source_commit"), str)
            or not isinstance(questions, list) or not questions
            or any(not isinstance(item, dict) or not isinstance(item.get("id"), str)
                   or not isinstance(item.get("question"), str)
                   or item.get("split") not in {"development", "test"}
                   or not isinstance(item.get("context_messages"), list)
                   or not item["context_messages"]
                   or any(not isinstance(message, dict)
                          or message.get("role") not in {"user", "assistant"}
                          or not isinstance(message.get("content"), str)
                          or not message["content"].strip()
                          for message in item["context_messages"])
                   or ("positive_entity_ids" not in item and "positive_passage_ids" not in item)
                   or ("positive_entity_ids" in item and (not isinstance(item["positive_entity_ids"], list)
                       or not item["positive_entity_ids"]
                       or any(not isinstance(value, str) for value in item["positive_entity_ids"])))
                   or ("positive_passage_ids" in item and (not isinstance(item["positive_passage_ids"], list)
                       or not item["positive_passage_ids"]
                       or not isinstance(item.get("expected_source_type"), str)
                       or not isinstance(item.get("expected_kind"), str)
                       or any(not isinstance(value, str) for value in item["positive_passage_ids"])))
                   for item in questions)):
        raise ValueError("The contextual retrieval benchmark dataset is invalid.")
    if len({item["id"] for item in questions}) != len(questions):
        raise ValueError("The contextual retrieval benchmark contains duplicate question IDs.")
    if {item["split"] for item in questions} != {"development", "test"}:
        raise ValueError("The contextual retrieval benchmark must include development and test cases.")
    return payload


def available_embedding_models(base_url: str) -> list[dict[str, Any]]:
    """Return installed Ollama models with query/document formats supported by this app."""
    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            response = client.get(base_url.rstrip("/") + "/api/tags")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ModelError("The local model service could not list installed models.") from exc
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ModelError("The local model service returned an invalid model list.")
    size_cap = max_model_size_bytes()
    result = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name, size = item.get("name"), item.get("size")
        if (not isinstance(name, str) or name.split(":", 1)[0] not in SUPPORTED_EMBEDDING_FAMILIES
                or not isinstance(size, int) or size > size_cap):
            continue
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        result.append({
            "name": name,
            "size_bytes": size if isinstance(size, int) else None,
            "size_gb": round(size / (1024 ** 3), 2) if isinstance(size, int) else None,
            "parameter_size": details.get("parameter_size"),
            "format": details.get("format"),
            "quantization": details.get("quantization_level"),
        })
    return sorted(result, key=lambda item: item["name"].casefold())


def _slug(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def score_retrieval(candidates: list[dict[str, Any]], positives: list[str]) -> dict[str, Any]:
    """Score labels among the top six passages, deduplicating by entity."""
    ranked = list(dict.fromkeys(item["entity_id"] for item in candidates[:6]
                                if isinstance(item.get("entity_id"), str)))
    gold = set(positives)
    if not gold:
        raise ValueError("A retrieval case must have at least one judged positive entity.")
    first = next((rank for rank, entity_id in enumerate(ranked, 1) if entity_id in gold), None)
    return {
        "ranked_entity_ids": ranked,
        "matched_entity_ids": [entity_id for entity_id in ranked if entity_id in gold],
        "hit_at_1": bool(first and first <= 1),
        "hit_at_3": bool(first and first <= 3),
        "hit_at_6": first is not None,
        "recall_at_6": len(gold.intersection(ranked)) / len(gold),
        "mrr_at_6": 1 / first if first else 0.0,
    }


def score_passage_retrieval(candidates: list[dict[str, Any]], positives: list[str]) -> dict[str, Any]:
    """Score exact source passages, retaining the indexed object type in the output."""
    ranked = list(dict.fromkeys(item["id"] for item in candidates[:6]
                                if isinstance(item.get("id"), str)))
    gold = set(positives)
    if not gold:
        raise ValueError("A catalog-object case must have at least one judged passage.")
    first = next((rank for rank, passage_id in enumerate(ranked, 1) if passage_id in gold), None)
    return {
        "ranked_passage_ids": ranked,
        "matched_passage_ids": [passage_id for passage_id in ranked if passage_id in gold],
        "ranked_passages": [{key: item.get(key) for key in
                              ("id", "source_id", "source_type", "kind", "source_label", "title")}
                             for item in candidates[:6]],
        "hit_at_1": bool(first and first <= 1),
        "hit_at_3": bool(first and first <= 3),
        "hit_at_6": first is not None,
        "recall_at_6": len(gold.intersection(ranked)) / len(gold),
        "mrr_at_6": 1 / first if first else 0.0,
    }


def _aggregate(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    measured = [row["variants"][mode] for row in rows]
    latencies = sorted(item["latency_ms"] for item in measured)
    if not measured:
        return {"cases": 0}
    return {
        "cases": len(measured),
        **{key: round(statistics.mean(float(item[key]) for item in measured), 4)
           for key in ("hit_at_1", "hit_at_3", "hit_at_6", "recall_at_6", "mrr_at_6")},
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "p95_latency_ms": round(latencies[math.ceil(0.95 * len(latencies)) - 1], 2),
    }


def append_retrieval_benchmark_run(record: dict[str, Any]) -> bool:
    # Successful writes should retain their persistence status when reopened.
    line = json.dumps({**record, "log_persisted": True}, ensure_ascii=False,
                      separators=(",", ":"), default=str)
    try:
        with _LOG_LOCK:
            RETRIEVAL_BENCHMARK_LOG_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with RETRIEVAL_BENCHMARK_LOG_PATH.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
    except OSError:
        return False
    return True


def list_saved_retrieval_runs(limit: int = MAX_SAVED_RUNS_IN_PAGE) -> list[dict[str, Any]]:
    if not RETRIEVAL_BENCHMARK_LOG_PATH.exists():
        return []
    recent: list[dict[str, Any]] = []
    try:
        with _LOG_LOCK, RETRIEVAL_BENCHMARK_LOG_PATH.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or not isinstance(record.get("run_id"), str):
                    continue
                summary = {key: record.get(key) for key in (
                    "run_id", "track", "started_at", "elapsed_seconds", "split", "model_names",
                    "graph_methods", "query_model", "query_methods", "query_method", "rerankers",
                    "query_models", "question_set", "question_ids", "summaries", "log_persisted", "source_commit",
                    "feature_model", "edge_count", "evaluation_edge_count", "split_counts")
                }
                if summary["log_persisted"] is None:
                    summary["log_persisted"] = True
                recent.append(summary)
                if len(recent) > limit:
                    del recent[0]
    except OSError as exc:
        raise ModelError("The local retrieval benchmark history could not be read from disk.") from exc
    return list(reversed(recent))


def load_saved_retrieval_run(run_id: str) -> dict[str, Any] | None:
    if not RETRIEVAL_BENCHMARK_LOG_PATH.exists():
        return None
    try:
        with _LOG_LOCK, RETRIEVAL_BENCHMARK_LOG_PATH.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get("run_id") == run_id:
                    record.setdefault("log_persisted", True)
                    return record
    except OSError as exc:
        raise ModelError("The saved retrieval benchmark result could not be read from disk.") from exc
    return None


def run_retrieval_benchmark(catalog_path: Path, data_dir: Path, configured_models: Ollama,
                            model_names: list[str], question_ids: list[str], split: str,
                            dataset: dict[str, Any], run_id: str, model_lock: threading.Lock,
                            emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Search the same labeled questions across embedding models; never generate or rerank."""
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if split not in {"development", "test", "complex"}:
        raise ValueError("Choose development, test, or complex questions.")
    if not model_names or len(model_names) > 6 or len(set(model_names)) != len(model_names):
        raise ValueError("Select between one and six distinct embedding models.")
    if not question_ids or len(question_ids) > 42 or len(set(question_ids)) != len(question_ids):
        raise ValueError("Select between one and 42 distinct questions.")
    cases_by_id = {item["id"]: item for item in dataset["questions"]}
    if any(item not in cases_by_id or cases_by_id[item]["split"] != split for item in question_ids):
        raise ValueError("Select questions from the chosen evaluation split.")
    cases = [cases_by_id[item] for item in question_ids]
    track = ("catalog_objects" if dataset.get("dataset_id") == "cdm-catalog-object-retrieval-v1"
             else "passages")
    installed = {item["name"] for item in available_embedding_models(configured_models.base_url)}
    if not set(model_names).issubset(installed):
        raise ValueError("Select only installed embedding models with supported query formats.")

    emit({"type": "retrieval_benchmark_started", "run_id": run_id, "track": track, "split": split,
          "model_names": model_names, "question_count": len(cases),
          "modes": ["lexical", "semantic", "hybrid"], "reranking": False,
          "generation": False})
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    with model_lock:
        for model_position, model_name in enumerate(model_names, 1):
            emit({"type": "retrieval_model_started", "run_id": run_id, "model": model_name,
                  "position": model_position, "total": len(model_names)})
            model = Ollama(configured_models.base_url, model_name,
                           configured_models.router_model, configured_models.answer_model,
                           configured_models.scope_model, configured_models.judge_model,
                           configured_models.request_timeout, configured_models.stream_idle_timeout,
                           configured_models.first_token_timeout)
            # Keep only this embedding model resident during its batch, then free it before
            # the next model or any answer-generation workload can run.
            model.query_embedding_keep_alive = "5m"
            index: SchemaIndex | None = None
            model_rows: list[dict[str, Any]] = []
            index_seconds = 0.0
            index_reused = False
            try:
                emit({"type": "retrieval_index_started", "run_id": run_id,
                      "model": model_name, "position": model_position,
                      "message": "Opening or building this model's local passage index."})
                index_started = time.perf_counter()
                index = SchemaIndex(catalog_path,
                                    data_dir / "retrieval_benchmark_indices" / _slug(model_name), model)
                index_seconds = round(time.perf_counter() - index_started, 2)
                index_reused = bool(getattr(index, "index_reused", False))
                if dataset.get("source_commit") != index.catalog.get("source_commit"):
                    raise ValueError("The retrieval questions and local catalog target different CDM revisions.")
                for case in cases:
                    if "positive_passage_ids" in case:
                        for passage_id in case["positive_passage_ids"]:
                            passage = index.by_id.get(passage_id)
                            if (passage is None or passage.get("source_type") != case["expected_source_type"]
                                    or passage.get("kind") != case["expected_kind"]):
                                raise ValueError(
                                    f"A labeled {case['expected_kind']} passage is missing from this catalog revision."
                                )
                    elif set(case["positive_entity_ids"]) - set(index.graph.entities):
                        raise ValueError("A labeled entity is missing from this catalog revision.")
                emit({"type": "retrieval_index_ready", "run_id": run_id, "model": model_name,
                      "index_preparation_seconds": index_seconds, "index_reused": index_reused,
                      "chunk_count": len(index.chunks), "model_digest": index.model_digest,
                      "embedding_format": model.embedding_format()})

                # Warm the embedding/query path; per-question latency excludes cold startup.
                index.search(cases[0]["question"], limit=20, mode="semantic")
                for question_position, question in enumerate(cases, 1):
                    measured: dict[str, Any] = {}
                    emit({"type": "retrieval_question_started", "run_id": run_id,
                          "model": model_name, "question_id": question["id"],
                          "question": question["question"], "position": question_position,
                          "total": len(cases)})
                    for mode in ("lexical", "semantic", "hybrid"):
                        emit({"type": "retrieval_search_started", "run_id": run_id,
                              "model": model_name, "question_id": question["id"], "mode": mode})
                        search_started = time.perf_counter()
                        candidates = index.search(question["question"], limit=20, mode=mode)
                        elapsed_ms = (time.perf_counter() - search_started) * 1000
                        scorer = score_passage_retrieval if "positive_passage_ids" in question else score_retrieval
                        positives = question.get("positive_passage_ids", question.get("positive_entity_ids", []))
                        measured[mode] = {**scorer(candidates[:6], positives),
                                          "latency_ms": round(elapsed_ms, 2)}
                    row = {
                        "model": model_name,
                        "question_id": question["id"],
                        "question": question["question"],
                        "category": question.get("category"),
                        "split": question["split"],
                        "object_type": question.get("object_type"),
                        "expected_source_type": question.get("expected_source_type"),
                        "expected_kind": question.get("expected_kind"),
                        "positive_entity_ids": question.get("positive_entity_ids"),
                        "positive_passage_ids": question.get("positive_passage_ids"),
                        "variants": measured,
                    }
                    rows.append(row)
                    model_rows.append(row)
                    emit({"type": "retrieval_row", "run_id": run_id, "row": row})
                runtime = model.runtime_info(model_name)
                summary = {
                    "model": model_name,
                    "question_count": len(model_rows),
                    "index_preparation_seconds": index_seconds,
                    "index_reused": index_reused,
                    "model_digest": index.model_digest,
                    "embedding_format": model.embedding_format(),
                    "runtime": runtime,
                    "metrics": {mode: _aggregate(model_rows, mode)
                                for mode in ("lexical", "semantic", "hybrid")},
                }
                summaries.append(summary)
                emit({"type": "retrieval_model_completed", "run_id": run_id,
                      "model": model_name, "summary": summary,
                      "position": model_position, "total": len(model_names)})
            finally:
                index = None
                try:
                    model.unload_if_loaded(model_name)
                    emit({"type": "retrieval_model_unloaded", "run_id": run_id, "model": model_name})
                except ModelError as exc:
                    emit({"type": "retrieval_unload_warning", "run_id": run_id,
                          "model": model_name, "detail": str(exc)})

    result = {
        "track": track,
        "run_id": run_id,
        "started_at": started_at,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "dataset_id": dataset["dataset_id"],
        "base_dataset_id": dataset.get("base_dataset_id"),
        "supplemental_dataset_id": dataset.get("supplemental_dataset_id"),
        "dataset_status": dataset["status"],
        "dataset_provenance": dataset.get("provenance"),
        "supplemental_provenance": dataset.get("supplemental_provenance"),
        "source_commit": dataset["source_commit"],
        "split": split,
        "model_names": model_names,
        "question_ids": question_ids,
        "candidate_limit": 20,
        "top_k_passages": 6,
        "reranking": False,
        "generation": False,
        "notes": ("Labels are incomplete and assistant-authored, not independently reviewed; unlisted passages and "
                  "entities are unjudged. "
                  + ("Catalog-object metrics rank exact labeled passage IDs among the first six retrieved passages. "
                     if dataset.get("dataset_id") == "cdm-catalog-object-retrieval-v1" else
                     "Ranking metrics deduplicate entities among the first six retrieved passages. ")
                  + "Semantic and hybrid "
                  "latencies are warm local searches and include query embedding, but not index preparation."),
        "summaries": summaries,
        "rows": rows,
    }
    result["log_persisted"] = append_retrieval_benchmark_run(result)
    emit({"type": "retrieval_benchmark_completed", "result": result})
    return result
