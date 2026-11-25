"""Controlled query-processing and reranking experiments for passage retrieval."""
from __future__ import annotations

import gc
import importlib.util
import math
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Callable

from .benchmark import available_chat_models
from .index import SchemaIndex
from .ollama import ModelError, Ollama
from .rag import plan_search_timed, rerank as llm_rerank, rewrite_search_query
from .retrieval_benchmark import (_aggregate, _slug, available_embedding_models,
                                  score_passage_retrieval, score_retrieval)


RERANKER_CATALOG = {
    "ollama_llm": {
        "label": "Ollama LLM reranker",
        "model": "Configured answer model",
        "description": "The current JSON-list reranker using the API answer model.",
        "requires": None,
    },
    "bce": {
        "label": "BCE cross-encoder",
        "model": "maidalun1020/bce-reranker-base_v1",
        "description": "Open-weight query-passage scorer; downloads on first use; PyTorch uses CUDA, Apple MPS, or CPU.",
        "requires": "sentence-transformers",
    },
    "bge": {
        "label": "BGE cross-encoder",
        "model": "BAAI/bge-reranker-v2-m3",
        "description": "Open-weight multilingual query-passage scorer; downloads on first use; PyTorch uses CUDA, Apple MPS, or CPU.",
        "requires": "sentence-transformers",
    },
}
QUERY_METHODS = {"raw", "planner", "rewrite"}
RERANKER_METHODS = set(RERANKER_CATALOG)


def reranker_options() -> list[dict[str, Any]]:
    # Do not import PyTorch just to render the options page; that can reserve
    # substantial memory even when the cross-encoder experiment is not used.
    cross_encoder_available = importlib.util.find_spec("sentence_transformers") is not None
    return [{
        "id": method,
        "label": details["label"],
        "model": details["model"],
        "description": details["description"],
        "available": method == "ollama_llm" or cross_encoder_available,
        "install_command": ("uv pip install -r requirements-reranker-benchmark.txt"
                            if details["requires"] and not cross_encoder_available else None),
    } for method, details in RERANKER_CATALOG.items()]


def _query_plan(method: str, question: str, models: Ollama,
                context_messages: list[dict[str, str]] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    if method == "raw":
        context = "\n".join(f"{item['role'].title()}: {item['content']}" for item in context_messages or [])
        raw_query = f"{context}\nUser: {question}" if context else question
        return {"query": raw_query, "route": "raw", "inference_ms": 0.0,
                "inference_timings": None, "error": None}
    try:
        if method == "planner":
            should_search, query, timings = plan_search_timed(question, models, context_messages)
            normalized_question = " ".join(question.split()).casefold()
            route = ("no_search" if not should_search else
                     "raw_search" if " ".join(query.split()).casefold() == normalized_question else
                     "rewrite_search")
            if not should_search:
                query = ""
            return {"query": query, "route": route,
                    "inference_ms": round((time.perf_counter() - started) * 1000, 2),
                    "inference_timings": timings, "error": None}
        query = rewrite_search_query(question, models, context_messages)
        return {"query": query, "route": "rewrite_only",
                "inference_ms": round((time.perf_counter() - started) * 1000, 2),
                "inference_timings": None, "error": None}
    except ModelError as exc:
        return {"query": "", "route": "inference_error",
                "inference_ms": round((time.perf_counter() - started) * 1000, 2),
                "inference_timings": None, "error": str(exc)}


def _cross_encoder_device(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _load_cross_encoder(method: str) -> tuple[Any, Any, str, float]:
    try:
        import torch
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ValueError("Cross-encoder reranking needs the optional sentence-transformers package. "
                         "Install it with `uv pip install -r requirements-reranker-benchmark.txt`.") from exc
    device = _cross_encoder_device(torch)
    model_name = RERANKER_CATALOG[method]["model"]
    model_kwargs = {"torch_dtype": torch.float16} if device in {"cuda", "mps"} else {}
    started = time.perf_counter()
    model = CrossEncoder(model_name, device=device, max_length=512,
                         model_kwargs=model_kwargs)
    return model, torch, device, round(time.perf_counter() - started, 3)


def _cross_encoder_rank(model: Any, query: str, candidates: list[dict[str, Any]],
                        limit: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    pairs = [(query, str(candidate.get("text", ""))) for candidate in candidates]
    scores = model.predict(pairs, batch_size=8, show_progress_bar=False,
                          convert_to_numpy=True)
    values = [float(value) for value in scores]
    if len(values) != len(candidates) or any(not math.isfinite(value) for value in values):
        raise ModelError("The cross-encoder returned invalid relevance scores.")
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    return [candidates[index] for index in order[:limit]]


def _release_cross_encoders(models: dict[str, tuple[Any, Any, str, float]]) -> None:
    runtimes = [(item[1], item[2]) for item in models.values()]
    models.clear()
    gc.collect()
    for torch, device in runtimes:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()


def run_query_retrieval_benchmark(
        catalog_path, data_dir, configured_models: Ollama, embedding_model: str,
        question_ids: list[str], split: str, dataset: dict[str, Any], run_id: str,
        model_lock, emit: Callable[[dict[str, Any]], None], persist: Callable[[dict[str, Any]], bool],
        *, experiment: str, query_model: str, query_methods: list[str] | None = None,
        query_method: str | None = None, rerankers: list[str] | None = None,
        query_models: list[str] | None = None,
        question_set: str = "entities") -> dict[str, Any]:
    """Compare query transformations/models or rerankers while holding the embedder fixed."""
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if experiment not in {"query_processing", "reranking"}:
        raise ValueError("Choose query processing or reranking for this benchmark run.")
    if split not in {"development", "test", "complex"}:
        raise ValueError("Choose development, test, or complex questions.")
    if question_set not in {"entities", "catalog_objects", "contextual"}:
        raise ValueError("Choose the entity, catalog-object, or contextual question set.")
    if question_set in {"catalog_objects", "contextual"} and split == "complex":
        raise ValueError("The selected question set has development and test splits only.")
    methods = list(query_methods or []) if experiment == "query_processing" else [query_method or ""]
    if not methods or any(method not in QUERY_METHODS for method in methods) or len(set(methods)) != len(methods):
        raise ValueError("Select valid, distinct query-processing methods.")
    selected_rerankers = list(dict.fromkeys(rerankers or []))
    if experiment == "reranking" and any(method not in RERANKER_METHODS for method in selected_rerankers):
        raise ValueError("Select only supported reranking methods.")
    if experiment == "query_processing" and selected_rerankers:
        raise ValueError("Query-processing experiments run without reranking.")
    if (not question_ids or len(question_ids) > 42 or len(set(question_ids)) != len(question_ids)):
        raise ValueError("Select between one and 42 distinct questions.")
    cases_by_id = {item["id"]: item for item in dataset["questions"]}
    if any(item not in cases_by_id or cases_by_id[item]["split"] != split for item in question_ids):
        raise ValueError("Select questions from the chosen evaluation split.")
    cases = [cases_by_id[item] for item in question_ids]
    installed_embeddings = {item["name"] for item in available_embedding_models(configured_models.base_url)}
    if embedding_model not in installed_embeddings:
        raise ValueError("Choose one installed embedding model and keep it fixed for this experiment.")
    installed_query_models = {item["name"] for item in available_chat_models(configured_models.base_url)}
    selected_query_models = (list(dict.fromkeys(query_models or [])) if experiment == "query_processing"
                             else [query_model])
    if experiment == "query_processing":
        needs_inference = any(method != "raw" for method in methods)
        if needs_inference and not selected_query_models:
            raise ValueError("Select at least one local model for query inference.")
        if not set(selected_query_models).issubset(installed_query_models):
            raise ValueError("Choose installed local query-inference models within the configured size cap.")
    elif query_model not in installed_query_models:
        raise ValueError("Choose one installed local query-inference model within the configured size cap.")
    if "ollama_llm" in selected_rerankers and configured_models.answer_model not in installed_query_models:
        raise ValueError("The configured answer model is not installed for LLM reranking.")
    if any(method in {"bce", "bge"} for method in selected_rerankers):
        reranker_options_by_id = {item["id"]: item for item in reranker_options()}
        if any(not reranker_options_by_id[method]["available"]
               for method in selected_rerankers if method in {"bce", "bge"}):
            raise ValueError("Install the optional sentence-transformers dependency before choosing BCE or BGE.")

    plan_specs: list[tuple[str, str | None]] = []
    if experiment == "query_processing":
        if "raw" in methods:
            plan_specs.append(("raw", None))
        for selected_model in selected_query_models:
            plan_specs.extend((method, selected_model) for method in methods if method != "raw")
    else:
        plan_specs.append((methods[0], query_model))
    plan_variants = [method if method == "raw" else f"{method}::{model_name}"
                     for method, model_name in plan_specs]
    plan_details = dict(zip(plan_variants, plan_specs))
    if not plan_specs:
        raise ValueError("Select at least one query creation method and a local model where needed.")

    emit({"type": "retrieval_benchmark_started", "run_id": run_id, "track": experiment,
          "split": split, "model_names": [embedding_model], "question_count": len(cases),
          "query_methods": methods if experiment == "query_processing" else [methods[0]],
          "query_models": selected_query_models,
          "rerankers": ["none", *selected_rerankers] if experiment == "reranking" else [],
          "generation": False})
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    cross_models: dict[str, tuple[Any, Any, str, float]] = {}
    with model_lock:
        model = Ollama(configured_models.base_url, embedding_model,
                       selected_query_models[0] if selected_query_models else query_model,
                       configured_models.answer_model, configured_models.scope_model,
                       configured_models.judge_model, configured_models.request_timeout,
                       configured_models.stream_idle_timeout, configured_models.first_token_timeout)
        model.query_embedding_keep_alive = "5m"
        index = None
        try:
            emit({"type": "retrieval_model_started", "run_id": run_id,
                  "model": embedding_model, "position": 1, "total": 1})
            index_started = time.perf_counter()
            index = SchemaIndex(catalog_path,
                                data_dir / "retrieval_benchmark_indices" / _slug(embedding_model), model)
            index_preparation_seconds = round(time.perf_counter() - index_started, 3)
            index_identity = index.index_revision
            index_reused = bool(getattr(index, "index_reused", False))
            if dataset.get("source_commit") != index.catalog.get("source_commit"):
                raise ValueError("The retrieval questions and local catalog target different CDM revisions.")
            for case in cases:
                if "positive_passage_ids" in case:
                    for passage_id in case["positive_passage_ids"]:
                        passage = index.by_id.get(passage_id)
                        if (passage is None or passage.get("source_type") != case["expected_source_type"]
                                or passage.get("kind") != case["expected_kind"]):
                            raise ValueError("A labeled catalog-object passage is missing from this catalog revision.")
                elif set(case.get("positive_entity_ids", [])) - set(index.graph.entities):
                    raise ValueError("A labeled entity is missing from this catalog revision.")
            emit({"type": "retrieval_index_ready", "run_id": run_id, "model": embedding_model,
                  "index_preparation_seconds": index_preparation_seconds,
                  "index_reused": index_reused, "chunk_count": len(index.chunks),
                  "model_digest": index.model_digest, "embedding_format": model.embedding_format()})

            # Clear resident Ollama models before loading the query model.
            for model_name in dict.fromkeys((embedding_model, *selected_query_models, query_model,
                                             configured_models.router_model,
                                             configured_models.answer_model,
                                             configured_models.scope_model,
                                             configured_models.judge_model)):
                try:
                    model.unload_if_loaded(model_name)
                except ModelError as exc:
                    emit({"type": "retrieval_unload_warning", "run_id": run_id,
                          "model": model_name, "detail": str(exc)})

            # Infer every search query before reloading the embedder for retrieval.
            query_plans: dict[str, dict[str, dict[str, Any]]] = {variant: {} for variant in plan_variants}
            previous_model = None
            for variant, (method, selected_model) in plan_details.items():
                if selected_model:
                    if previous_model and previous_model != selected_model:
                        try:
                            model.unload_if_loaded(previous_model)
                        except ModelError as exc:
                            emit({"type": "retrieval_unload_warning", "run_id": run_id,
                                  "model": previous_model, "detail": str(exc)})
                    model.router_model = selected_model
                for position, case in enumerate(cases, 1):
                    emit({"type": "retrieval_query_inference_started", "run_id": run_id,
                          "model": embedding_model, "query_method": method,
                          "query_model": selected_model, "query_variant": variant,
                          "question_id": case["id"],
                          "position": position, "total": len(cases)})
                    plan = _query_plan(method, case["question"], model, case.get("context_messages"))
                    query_plans[variant][case["id"]] = plan
                    emit({"type": "retrieval_query_inference_completed", "run_id": run_id,
                          "query_method": method, "query_model": selected_model,
                          "query_variant": variant,
                          "question_id": case["id"], "query": plan["query"],
                          "route": plan["route"], "inference_ms": plan["inference_ms"],
                          "error": plan["error"]})
                previous_model = selected_model or previous_model
            for selected_model in dict.fromkeys(selected_query_models):
                try:
                    model.unload_if_loaded(selected_model)
                except ModelError as exc:
                    emit({"type": "retrieval_unload_warning", "run_id": run_id,
                          "model": selected_model, "detail": str(exc)})
            # Keep the first search from absorbing the embedder's cold-load cost.
            # The one-time load is recorded separately so every query variant
            # receives the same warm retrieval path.
            emit({"type": "retrieval_embedding_warmup_started", "run_id": run_id,
                  "model": embedding_model})
            embedding_warmup_started = time.perf_counter()
            model.embed(["CDM retrieval benchmark warm-up"], query=True)
            embedding_warmup_ms = round((time.perf_counter() - embedding_warmup_started) * 1000, 2)
            emit({"type": "retrieval_embedding_warmup_completed", "run_id": run_id,
                  "model": embedding_model, "warmup_ms": embedding_warmup_ms})
            # Retrieve against one fixed index for every query variant.
            retrieved: dict[tuple[str, str], tuple[list[dict[str, Any]], float]] = {}
            for position, case in enumerate(cases, 1):
                emit({"type": "retrieval_question_started", "run_id": run_id,
                      "model": embedding_model, "question_id": case["id"],
                      "question": case["question"], "position": position, "total": len(cases)})
                for variant in plan_variants:
                    plan = query_plans[variant][case["id"]]
                    query = plan["query"]
                    emit({"type": "retrieval_search_started", "run_id": run_id,
                          "model": embedding_model, "question_id": case["id"],
                          "mode": variant})
                    if query:
                        search_started = time.perf_counter()
                        candidates = index.search(query, limit=20, mode="hybrid")
                        search_ms = round((time.perf_counter() - search_started) * 1000, 2)
                    else:
                        candidates, search_ms = [], 0.0
                    retrieved[(variant, case["id"])] = (candidates, search_ms)

            try:
                model.unload_if_loaded(embedding_model)
            except ModelError as exc:
                emit({"type": "retrieval_unload_warning", "run_id": run_id,
                      "model": embedding_model, "detail": str(exc)})

            # Chroma and BM25 can be released before loading optional reranker weights.
            index = None
            gc.collect()
            try:
                model.unload_if_loaded(configured_models.answer_model)
            except ModelError as exc:
                emit({"type": "retrieval_unload_warning", "run_id": run_id,
                      "model": configured_models.answer_model, "detail": str(exc)})

            reranker_load_seconds: dict[str, float] = {}
            reranker_load_errors: dict[str, str] = {}
            reranker_devices: dict[str, str] = {}
            summary_rows: dict[str, list[dict[str, Any]]] = {}
            inference_by_variant: dict[str, list[float]] = {}
            search_by_variant: dict[str, list[float]] = {}
            rerank_by_variant: dict[str, list[float]] = {}
            result_variants = plan_variants if experiment == "query_processing" else ["none", *selected_rerankers]
            for variant in result_variants:
                summary_rows[variant] = []
                inference_by_variant[variant] = []
                search_by_variant[variant] = []
                rerank_by_variant[variant] = []
            model_rows = [{"model": embedding_model, "question_id": case["id"],
                           "question": case["question"], "category": case.get("category"),
                           "split": case["split"], "object_type": case.get("object_type"),
                           "context_messages": case.get("context_messages"),
                           "expected_source_type": case.get("expected_source_type"),
                           "expected_kind": case.get("expected_kind"),
                           "positive_entity_ids": case.get("positive_entity_ids"),
                           "positive_passage_ids": case.get("positive_passage_ids"),
                           "query_method": methods[0] if experiment == "reranking" else None,
                           "query": query_plans[plan_variants[0]][case["id"]]["query"]
                           if experiment == "reranking" else None,
                           "variants": {}, "query_plans": {}} for case in cases]

            for variant in result_variants:
                method, variant_query_model = (plan_details[variant] if experiment == "query_processing"
                                               else plan_details[plan_variants[0]])
                plan_variant = variant if experiment == "query_processing" else plan_variants[0]
                if experiment == "reranking" and variant in {"bce", "bge"}:
                    emit({"type": "retrieval_reranker_loading", "run_id": run_id,
                          "reranker": variant, "model": RERANKER_CATALOG[variant]["model"]})
                    load_started = time.perf_counter()
                    try:
                        cross_models[variant] = _load_cross_encoder(variant)
                        reranker_load_seconds[variant] = round(time.perf_counter() - load_started, 3)
                        reranker_devices[variant] = cross_models[variant][2]
                    except Exception as exc:
                        reranker_load_errors[variant] = (
                            f"Could not load the {RERANKER_CATALOG[variant]['label']} model. "
                            "Check network access and available memory, then retry.")
                        emit({"type": "retrieval_reranker_load_failed", "run_id": run_id,
                              "reranker": variant, "detail": reranker_load_errors[variant]})

                for position, case in enumerate(cases, 1):
                    plan = query_plans[plan_variant][case["id"]]
                    candidates, search_ms = retrieved[(plan_variant, case["id"])]
                    if experiment == "query_processing":
                        selected, error, rerank_ms = candidates[:6], plan["error"], 0.0
                    else:
                        emit({"type": "retrieval_reranking_started", "run_id": run_id,
                              "reranker": variant, "question_id": case["id"],
                              "position": position, "total": len(cases)})
                        rerank_started = time.perf_counter()
                        error = None
                        try:
                            if variant == "none":
                                selected = candidates[:6]
                            elif variant == "ollama_llm":
                                selected = llm_rerank(plan["query"], candidates, model, 6)
                            elif variant in reranker_load_errors:
                                selected, error = [], reranker_load_errors[variant]
                            else:
                                selected = _cross_encoder_rank(cross_models[variant][0],
                                                               plan["query"], candidates, 6)
                        except Exception as exc:
                            selected = []
                            error = str(exc) if isinstance(exc, (ModelError, ValueError)) else "Reranker scoring failed."
                        rerank_ms = round((time.perf_counter() - rerank_started) * 1000, 2)
                        emit({"type": "retrieval_reranking_completed", "run_id": run_id,
                              "reranker": variant, "question_id": case["id"],
                              "rerank_ms": rerank_ms, "error": error})

                    scorer = score_passage_retrieval if "positive_passage_ids" in case else score_retrieval
                    positives = case.get("positive_passage_ids", case.get("positive_entity_ids", []))
                    measured = {**scorer(selected, positives),
                                "query": plan["query"], "route": plan["route"],
                                "reranker": variant if experiment == "reranking" else None,
                                "query_inference_ms": plan["inference_ms"],
                                "search_ms": search_ms, "rerank_ms": rerank_ms,
                                "latency_ms": plan["inference_ms"] + search_ms + rerank_ms,
                                "error": error or plan["error"]}
                    model_rows[position - 1]["variants"][variant] = measured
                    model_rows[position - 1]["query_plans"][variant] = plan
                    summary_rows[variant].append({"variants": {variant: measured}})
                    inference_by_variant[variant].append(plan["inference_ms"])
                    search_by_variant[variant].append(search_ms)
                    rerank_by_variant[variant].append(rerank_ms)

                if variant in {"bce", "bge"} and cross_models:
                    _release_cross_encoders(cross_models)
                if variant == "ollama_llm":
                    try:
                        model.unload_if_loaded(configured_models.answer_model)
                    except ModelError as exc:
                        emit({"type": "retrieval_unload_warning", "run_id": run_id,
                              "model": configured_models.answer_model, "detail": str(exc)})

            rows.extend(model_rows)
            for row in model_rows:
                emit({"type": "retrieval_row", "run_id": run_id, "row": row})

            metrics = {variant: _aggregate(summary_rows[variant], variant)
                       for variant in result_variants}
            timing_summary = {variant: {
                "mean_query_inference_ms": round(statistics.mean(inference_by_variant[variant]), 2),
                "mean_search_ms": round(statistics.mean(search_by_variant[variant]), 2),
                "mean_rerank_ms": round(statistics.mean(rerank_by_variant[variant]), 2),
                "mean_end_to_end_ms": round(metrics[variant]["mean_latency_ms"], 2),
            } for variant in result_variants}
            summary = {
                "model": embedding_model, "embedding_model": embedding_model,
                "question_count": len(model_rows), "index_identity": index_identity,
                "index_preparation_seconds": index_preparation_seconds,
                "index_reused": index_reused,
                "embedding_warmup_ms": embedding_warmup_ms,
                "query_model": query_model if experiment == "reranking" else None,
                "query_models": selected_query_models,
                "question_set": question_set,
                "query_model_by_variant": {variant: details[1] for variant, details in plan_details.items()},
                "query_methods": methods if experiment == "query_processing" else [methods[0]],
                "query_method": methods[0] if experiment == "reranking" else None,
                "rerankers": ["none", *selected_rerankers] if experiment == "reranking" else [],
                "reranker_models": {"ollama_llm": configured_models.answer_model,
                                    **{method: RERANKER_CATALOG[method]["model"]
                                       for method in selected_rerankers if method in RERANKER_CATALOG}},
                "reranker_load_seconds": reranker_load_seconds,
                "reranker_load_errors": reranker_load_errors,
                "reranker_devices": reranker_devices,
                "runtime": model.runtime_info(embedding_model),
                "metrics": metrics, "timings": timing_summary,
            }
            summaries.append(summary)
            emit({"type": "retrieval_model_completed", "run_id": run_id,
                  "model": embedding_model, "summary": summary, "position": 1, "total": 1})
        finally:
            _release_cross_encoders(cross_models)
            for model_name in dict.fromkeys((*selected_query_models, query_model, embedding_model,
                                             configured_models.router_model,
                                             configured_models.answer_model,
                                             configured_models.scope_model,
                                             configured_models.judge_model)):
                try:
                    model.unload_if_loaded(model_name)
                    emit({"type": "retrieval_model_unloaded", "run_id": run_id, "model": model_name})
                except ModelError as exc:
                    emit({"type": "retrieval_unload_warning", "run_id": run_id,
                          "model": model_name, "detail": str(exc)})

    result = {
        "track": experiment, "run_id": run_id, "started_at": started_at,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "dataset_id": dataset["dataset_id"], "source_commit": dataset["source_commit"],
        "dataset_status": dataset["status"], "dataset_provenance": dataset.get("provenance"),
        "split": split, "model_names": [embedding_model], "question_ids": question_ids,
        "question_set": question_set,
        "candidate_limit": 20, "top_k_passages": 6, "retrieval_mode": "hybrid",
        "query_model": query_model if experiment == "reranking" else None,
        "query_models": selected_query_models,
        "query_methods": methods if experiment == "query_processing" else [methods[0]],
        "query_method": methods[0] if experiment == "reranking" else None,
        "rerankers": ["none", *selected_rerankers] if experiment == "reranking" else [],
        "reranking": experiment == "reranking", "generation": False,
        "notes": ("One embedding model is held fixed for this run. Query-processing runs compare raw user wording, "
                  "the local planner, and a rewrite-only query against the same hybrid retrieval index; reranking "
                  "is disabled in this track. Contextual cases pass the same prior messages and current question "
                  "to raw search, planner, and rewrite-only variants. The query embedder is warmed once before "
                  "per-query retrieval timings, and that cost is recorded separately. Reranking runs hold one query strategy and model fixed, then compare "
                  "the no-rerank baseline with selected rerankers over the same top-20 hybrid candidates. "
                  "Candidate labels are incomplete, assistant-authored, and not independently reviewed."),
        "summaries": summaries, "rows": rows,
    }
    result["log_persisted"] = persist(result)
    emit({"type": "retrieval_benchmark_completed", "result": result})
    return result
