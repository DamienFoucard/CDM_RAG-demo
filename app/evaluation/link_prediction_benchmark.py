"""Held-out link prediction for the typed CDM graph projection."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.schema.corpus import load_catalog
from app.schema.graph import SchemaGraph
from app.evaluation.graph_retrieval_benchmark import (
    DIMENSIONS,
    EPOCHS,
    GRAPH_METHOD_LABELS,
    SEED,
    _normalize,
    _train_boxe,
    _train_gcn,
    _train_transe,
    _train_transh,
)
from app.runtime.ollama import ModelError, Ollama
from app.evaluation.retrieval_benchmark import available_embedding_models

LINK_PREDICTION_METHODS = ("gcn_1", "gcn_2", "gcn_3", "transe", "transh", "boxe")
LINK_PREDICTION_METHOD_LABELS = {key: value for key, value in GRAPH_METHOD_LABELS.items()
                                 if key in LINK_PREDICTION_METHODS}
LINK_PREDICTION_DATASET_ID = "cdm-typed-graph-link-prediction-v1"
FEATURE_CACHE_DIR = "link_prediction_indices/node_features"


def _node_text(node: dict[str, Any], kind: str) -> str:
    """Describe a node without including any of the graph edges being predicted."""
    name = str(node.get("name") or node.get("id") or "")
    if kind == "entity":
        attributes = ", ".join(str(item.get("name", "")) for item in node.get("attributes", [])[:20]
                                if isinstance(item, dict))
        return f"Entity {name}. {node.get('description', '')}. Attributes: {attributes}"
    if kind == "trait":
        return f"CDM trait {name}. {node.get('explanation', '')}. {node.get('verb', '')}"
    if kind == "manifest":
        return (f"CDM manifest {name}. {node.get('explanation', '')}. "
                f"Model: {node.get('model', '')}. Document version: {node.get('document_version', '')}. "
                f"Schema version: {node.get('json_schema_semantic_version', '')}.")
    return f"CDM base type {name}."


def _link_graph(graph: SchemaGraph) -> tuple[list[str], list[dict[str, Any]], np.ndarray,
                                               list[np.ndarray], list[dict[str, Any]], list[str]]:
    """Return typed nodes and unique triples from explicit graph layers."""
    nodes: dict[str, dict[str, Any]] = {}
    for entity in graph.entities.values():
        nodes[entity["id"]] = {**entity, "node_kind": "entity",
                                "candidate_group": f"entity:{entity.get('model', '')}"}
    layers = graph.semantic_layers()
    for kind in ("manifest", "trait", "base_type"):
        section = "inheritance" if kind == "base_type" else ("traits" if kind == "trait" else "manifests")
        for node in layers.get(section, {}).get("nodes", []):
            if isinstance(node, dict) and isinstance(node.get("id"), str):
                nodes[node["id"]] = {**node, "node_kind": kind, "candidate_group": kind}

    edge_records: list[dict[str, Any]] = []
    def add_edge(edge: dict[str, Any], relation: str, kind: str) -> None:
        source, target = edge.get("source_id"), edge.get("target_id")
        if source not in nodes or target not in nodes:
            return
        edge_records.append({"id": str(edge.get("id") or f"{kind}:{source}:{relation}:{target}"),
                             "source_id": source, "target_id": target,
                             "relation": relation, "edge_kind": kind})

    for edge in graph.edges.values():
        add_edge(edge, "relationship:" + str(edge.get("attribute", "relationship")), "entity_relationship")
    manifests = layers.get("manifests", {})
    for edge in manifests.get("edges", []):
        if isinstance(edge, dict):
            add_edge(edge, "manifest:" + str(edge.get("type", "structure")), "manifest_structure")
    for edge in manifests.get("relationships", []):
        if isinstance(edge, dict):
            add_edge(edge, "manifest_relationship", "manifest_relationship")
    traits = layers.get("traits", {})
    for edge in traits.get("edges", []):
        if isinstance(edge, dict):
            add_edge(edge, "trait_application", "trait_application")
    for edge in traits.get("inheritance_edges", []):
        if isinstance(edge, dict):
            add_edge(edge, "extends_trait", "trait_inheritance")
    inheritance = layers.get("inheritance", {})
    for edge in inheritance.get("edges", []):
        if isinstance(edge, dict):
            add_edge(edge, "extends_entity", "entity_inheritance")

    node_ids = sorted(nodes)
    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    relation_names = sorted({edge["relation"] for edge in edge_records})
    relation_to_index = {name: index for index, name in enumerate(relation_names)}
    unique: dict[tuple[int, int, int], dict[str, Any]] = {}
    for edge in sorted(edge_records, key=lambda item: item["id"]):
        triple = (node_to_index[edge["source_id"]], relation_to_index[edge["relation"]],
                  node_to_index[edge["target_id"]])
        unique.setdefault(triple, edge)
    ordered = sorted(unique.items())
    triples = np.asarray([triple for triple, _ in ordered], dtype=np.int64)
    edges = [edge for _, edge in ordered]
    pools_by_group: dict[str, np.ndarray] = {}
    for group in {node["candidate_group"] for node in nodes.values()}:
        pools_by_group[group] = np.asarray([node_to_index[node_id] for node_id in node_ids
                                            if nodes[node_id]["candidate_group"] == group], dtype=np.int64)
    pools = [pools_by_group[nodes[node_id]["candidate_group"]] for node_id in node_ids]
    return node_ids, [nodes[node_id] for node_id in node_ids], triples, pools, edges, relation_names


def _edge_splits(triples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratify by relation; every held-out relation retains at least one training edge."""
    rng = np.random.default_rng(SEED + 90)
    train: list[int] = []
    development: list[int] = []
    test: list[int] = []
    for relation in sorted(set(int(value) for value in triples[:, 1])):
        rows = np.flatnonzero(triples[:, 1] == relation)
        rng.shuffle(rows)
        count = len(rows)
        if count < 3:
            train.extend(int(value) for value in rows)
            continue
        if count == 3:
            train_count, dev_count = 1, 1
        else:
            train_count = max(1, int(math.floor(count * 0.8)))
            dev_count = max(1, int(round(count * 0.1)))
            if train_count + dev_count >= count:
                train_count = count - 2
        test_count = count - train_count - dev_count
        train.extend(int(value) for value in rows[:train_count])
        development.extend(int(value) for value in rows[train_count:train_count + dev_count])
        test.extend(int(value) for value in rows[train_count + dev_count:train_count + dev_count + test_count])
    return tuple(np.asarray(values, dtype=np.int64) for values in (train, development, test))


def link_prediction_graph_summary(catalog_path: Path) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    graph = SchemaGraph(catalog["entities"], catalog["source_commit"],
                        semantics=catalog.get("semantics", {}))
    node_ids, _, triples, _, edges, _ = _link_graph(graph)
    train, development, test = _edge_splits(triples)
    return {"dataset_id": LINK_PREDICTION_DATASET_ID, "source_commit": graph.revision,
            "graph_revision": graph.projection_revision, "node_count": len(node_ids),
            "edge_count": len(triples), "split_counts": {"train": len(train),
            "development": len(development), "test": len(test)},
            "edge_kind_counts": {kind: sum(edge["edge_kind"] == kind for edge in edges)
                                 for kind in sorted({edge["edge_kind"] for edge in edges})}}


def _feature_vectors(model: Ollama, nodes: list[dict[str, Any]], node_ids: list[str],
                     graph_revision: str, data_dir: Path,
                     emit: Callable[[dict[str, Any]], None], run_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    docs = [_node_text(node, node["node_kind"]) for node in nodes]
    digest = model.embedding_digest()
    identity = hashlib.sha256(json.dumps({"graph": graph_revision, "model": model.embedding_model,
                                          "digest": digest, "format": model.embedding_format(),
                                          "node_ids": node_ids, "docs": docs},
                                         sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache_dir = data_dir / FEATURE_CACHE_DIR
    cache_path = cache_dir / f"{identity}.npy"
    started = time.perf_counter()
    if cache_path.exists():
        try:
            vectors = np.load(cache_path, allow_pickle=False)
            if vectors.ndim == 2 and vectors.shape[0] == len(nodes):
                return vectors.astype(np.float32, copy=False), {
                    "identity": identity, "build_seconds": None,
                    "preparation_seconds": round(time.perf_counter() - started, 3),
                    "reused": True, "model_digest": digest,
                    "embedding_format": model.embedding_format(),
                }
        except (OSError, ValueError):
            cache_path.unlink(missing_ok=True)
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    build_started = time.perf_counter()
    vectors: list[list[float]] = []
    for offset in range(0, len(docs), 24):
        batch = docs[offset:offset + 24]
        emit({"type": "link_prediction_features", "run_id": run_id,
              "model": model.embedding_model, "completed": min(offset + len(batch), len(docs)),
              "total": len(docs)})
        vectors.extend(model.embed(batch, query=False, keep_alive="5m"))
    features = _normalize(np.asarray(vectors, dtype=np.float32))
    temporary = cache_path.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, features, allow_pickle=False)
    temporary.replace(cache_path)
    return features, {"identity": identity,
                      "build_seconds": round(time.perf_counter() - build_started, 3),
                      "preparation_seconds": round(time.perf_counter() - started, 3),
                      "reused": False, "model_digest": digest,
                      "embedding_format": model.embedding_format()}


def _scores(method: str, state: dict[str, np.ndarray], triples: np.ndarray) -> np.ndarray:
    heads, relations, tails = triples.T
    entities = state.get("entities")
    if method == "transe":
        return -np.abs(entities[heads] + state["relations"][relations] - entities[tails]).sum(axis=1)
    if method == "transh":
        normals, translations = state["normals"][relations], state["translations"][relations]
        delta = entities[heads] - entities[tails]
        projection = np.sum(delta * normals, axis=1, keepdims=True)
        residual = delta - projection * normals + translations
        return -np.abs(residual).sum(axis=1)
    if method == "boxe":
        entities, bumps = state["entities"], state["bumps"]
        centers, widths = state["centers"][relations], state["widths"][relations]
        head_points = entities[heads] + bumps[tails]
        tail_points = entities[tails] + bumps[heads]
        head_distance = _box_distance_batch(head_points, centers[:, 0], widths[:, 0])
        tail_distance = _box_distance_batch(tail_points, centers[:, 1], widths[:, 1])
        return -(head_distance + tail_distance)
    if method.startswith("gcn_"):
        vectors, rel = state["node_embeddings"], state["relations"]
        return np.sum(vectors[heads] * rel[relations] * vectors[tails], axis=1)
    raise ValueError(f"Unsupported link-prediction method: {method}")


def _box_distance_batch(points: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
    delta = points - centers
    absolute = np.abs(delta)
    inside = absolute <= ((widths - 1.0) / 2.0)
    kappa = 0.5 * (widths - 1.0) * (widths - 1.0 / widths)
    distance = np.where(inside, absolute / widths, absolute * widths - kappa)
    return distance.sum(axis=1)


def _filtered_rank(method: str, state: dict[str, np.ndarray], triple: np.ndarray, side: int,
                   pools: list[np.ndarray], known: set[tuple[int, int, int]]) -> float:
    candidates = np.repeat(triple[None, :], len(pools[int(triple[0 if side == 0 else 2])]), axis=0)
    candidate_nodes = pools[int(triple[0 if side == 0 else 2])]
    candidates[:, side] = candidate_nodes
    scores = _scores(method, state, candidates)
    target = tuple(int(value) for value in triple)
    for index, candidate in enumerate(candidates):
        candidate_key = tuple(int(value) for value in candidate)
        if candidate_key != target and candidate_key in known:
            scores[index] = -np.inf
    target_score = _scores(method, state, triple.reshape(1, 3))[0]
    greater = int(np.sum(scores > target_score))
    tied = int(np.sum(scores == target_score))
    return 1.0 + greater + max(tied - 1, 0) / 2.0


def _evaluate(method: str, state: dict[str, np.ndarray], indices: np.ndarray,
              triples: np.ndarray, pools: list[np.ndarray], known: set[tuple[int, int, int]],
              node_ids: list[str], nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    all_ranks: list[float] = []
    head_ranks: list[float] = []
    tail_ranks: list[float] = []
    for index in indices:
        triple = triples[int(index)]
        head_rank = _filtered_rank(method, state, triple, 0, pools, known)
        tail_rank = _filtered_rank(method, state, triple, 2, pools, known)
        head_ranks.append(head_rank)
        tail_ranks.append(tail_rank)
        all_ranks.extend((head_rank, tail_rank))
        edge = edges[int(index)]
        source, target = node_ids[int(triple[0])], node_ids[int(triple[2])]
        rows.append({"edge_id": edge["id"], "edge_kind": edge["edge_kind"],
                     "relation": edge["relation"], "source_id": source,
                     "source": nodes[int(triple[0])].get("name", source),
                     "source_kind": nodes[int(triple[0])]["node_kind"],
                     "target_id": target, "target": nodes[int(triple[2])].get("name", target),
                     "target_kind": nodes[int(triple[2])]["node_kind"],
                     "head_rank": round(head_rank, 4), "tail_rank": round(tail_rank, 4),
                     "reciprocal_rank": round((1.0 / head_rank + 1.0 / tail_rank) / 2.0, 6)})
    if not all_ranks:
        raise ValueError("The selected split has no held-out edges that can be ranked.")
    metrics = {
        "edge_count": len(rows),
        "prediction_count": len(all_ranks),
        "mrr": round(statistics.mean(1.0 / rank for rank in all_ranks), 4),
        "mean_rank": round(statistics.mean(all_ranks), 2),
        "hits_at_1": round(statistics.mean(rank <= 1 for rank in all_ranks), 4),
        "hits_at_3": round(statistics.mean(rank <= 3 for rank in all_ranks), 4),
        "hits_at_10": round(statistics.mean(rank <= 10 for rank in all_ranks), 4),
        "head_mrr": round(statistics.mean(1.0 / rank for rank in head_ranks), 4),
        "tail_mrr": round(statistics.mean(1.0 / rank for rank in tail_ranks), 4),
    }
    return metrics, rows


def run_link_prediction_benchmark(catalog_path: Path, data_dir: Path, configured_models: Ollama,
                                  feature_model_name: str, methods: list[str], split: str,
                                  run_id: str, model_lock,
                                  emit: Callable[[dict[str, Any]], None], persist) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if split not in {"development", "test"}:
        raise ValueError("Choose the development or test link-prediction split.")
    if not methods or len(set(methods)) != len(methods) or not set(methods).issubset(LINK_PREDICTION_METHODS):
        raise ValueError("Select one or more supported graph embedding methods.")
    needs_features = any(method.startswith("gcn_") for method in methods)
    if needs_features:
        installed = {item["name"] for item in available_embedding_models(configured_models.base_url)}
        if feature_model_name not in installed:
            raise ValueError("Choose one installed embedding model to provide GCN node features.")

    catalog = load_catalog(catalog_path)
    graph = SchemaGraph(catalog["entities"], catalog["source_commit"],
                        semantics=catalog.get("semantics", {}))
    node_ids, nodes, triples, pools, edges, relation_names = _link_graph(graph)
    if len(triples) < 12:
        raise ValueError("The projected CDM graph is too small for a held-out link-prediction evaluation.")
    train_indices, dev_indices, test_indices = _edge_splits(triples)
    eval_indices = dev_indices if split == "development" else test_indices
    fit_indices = train_indices if split == "development" else np.concatenate((train_indices, dev_indices))
    if not len(eval_indices):
        raise ValueError("There are no edges in this evaluation split; keep more edges in the graph revision.")
    train_triples = triples[fit_indices]
    # Negative sampling must not reveal held-out positives to the optimizer.
    # Keep the complete known set only for standard filtered ranking at eval.
    training_known = {tuple(int(value) for value in triple) for triple in train_triples}
    known = {tuple(int(value) for value in triple) for triple in triples}
    train_edges = [edges[int(index)] for index in fit_indices]
    entity_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    emit({"type": "link_prediction_benchmark_started", "run_id": run_id, "track": "link_prediction",
          "split": split, "methods": methods, "feature_model": feature_model_name if needs_features else None,
          "node_count": len(nodes), "edge_count": len(triples),
          "train_edge_count": len(train_indices), "development_edge_count": len(dev_indices),
          "test_edge_count": len(test_indices), "evaluation_edge_count": len(eval_indices)})

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    graph_training: dict[str, Any] = {}
    with model_lock:
        node_features = None
        feature_info: dict[str, Any] = {"build_seconds": 0.0, "preparation_seconds": 0.0,
                                        "reused": True, "embedding_model": None}
        if needs_features:
            model = Ollama(configured_models.base_url, feature_model_name,
                           configured_models.router_model, configured_models.answer_model,
                           configured_models.scope_model, configured_models.judge_model,
                           configured_models.request_timeout, configured_models.stream_idle_timeout,
                           configured_models.first_token_timeout)
            emit({"type": "link_prediction_feature_model_started", "run_id": run_id,
                  "model": feature_model_name, "node_count": len(nodes)})
            try:
                node_features, feature_info = _feature_vectors(
                    model, nodes, node_ids, graph.projection_revision, data_dir, emit, run_id)
                feature_info["embedding_model"] = feature_model_name
                emit({"type": "link_prediction_features_ready", "run_id": run_id,
                      "model": feature_model_name, "node_count": len(nodes), **feature_info})
            finally:
                try:
                    model.unload_if_loaded(feature_model_name)
                    emit({"type": "link_prediction_feature_model_unloaded", "run_id": run_id,
                          "model": feature_model_name})
                except ModelError as exc:
                    emit({"type": "link_prediction_unload_warning", "run_id": run_id,
                          "model": feature_model_name, "detail": str(exc)})

        trainer_by_method = {"transe": _train_transe, "transh": _train_transh, "boxe": _train_boxe}
        for position, method in enumerate(methods, 1):
            emit({"type": "link_prediction_method_started", "run_id": run_id,
                  "method": method, "label": LINK_PREDICTION_METHOD_LABELS[method],
                  "position": position, "total": len(methods), "training_edges": len(train_triples)})
            fit_started = time.perf_counter()
            if method.startswith("gcn_"):
                _, info = _train_gcn(method, node_features, train_triples, len(relation_names),
                                     pools, training_known, train_edges, entity_to_index,
                                     return_state=True)
            else:
                _, info = trainer_by_method[method](train_triples, len(nodes), len(relation_names),
                                                    pools, training_known, return_state=True)
            fit_elapsed = round(time.perf_counter() - fit_started, 6)
            state = info.pop("_model_state")
            evaluation_started = time.perf_counter()
            metrics, method_rows = _evaluate(method, state, eval_indices, triples, pools,
                                             known, node_ids, nodes, edges)
            evaluation_seconds = round(time.perf_counter() - evaluation_started, 6)
            clean_info = {**info, "fit_seconds": fit_elapsed,
                          "evaluation_seconds": evaluation_seconds,
                          "feature_preparation_seconds": feature_info.get("preparation_seconds", 0.0)
                          if method.startswith("gcn_") else 0.0,
                          "feature_build_seconds": feature_info.get("build_seconds")
                          if method.startswith("gcn_") else None,
                          "feature_index_reused": feature_info.get("reused")
                          if method.startswith("gcn_") else None,
                          "feature_model": feature_model_name if method.startswith("gcn_") else None}
            graph_training[method] = clean_info
            summary = {"method": method, "label": LINK_PREDICTION_METHOD_LABELS[method],
                       "feature_model": feature_model_name if method.startswith("gcn_") else None,
                       "training_edge_count": len(train_triples), "evaluation_edge_count": len(eval_indices),
                       "metrics": metrics, **clean_info}
            summaries.append(summary)
            for row in method_rows:
                rows.append({"method": method, **row})
            emit({"type": "link_prediction_method_completed", "run_id": run_id,
                  "method": method, "summary": summary, "position": position, "total": len(methods)})
            del state

    result = {
        "track": "link_prediction", "run_id": run_id, "started_at": started_at,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "dataset_id": LINK_PREDICTION_DATASET_ID, "dataset_status": "provisional",
        "dataset_provenance": "Deterministic relation-stratified split of the pinned typed CDM graph projection.",
        "source_commit": graph.revision, "graph_revision": graph.projection_revision,
        "split": split, "model_names": [feature_model_name] if needs_features else [],
        "feature_model": feature_model_name if needs_features else None,
        "graph_methods": methods, "question_ids": [],
        "node_count": len(nodes), "edge_count": len(triples),
        "edge_kind_counts": {kind: sum(edge["edge_kind"] == kind for edge in edges)
                             for kind in sorted({edge["edge_kind"] for edge in edges})},
        "split_counts": {"train": len(train_indices), "development": len(dev_indices),
                         "test": len(test_indices), "evaluated": len(eval_indices)},
        "training_edge_count": len(train_triples), "evaluation_edge_count": len(eval_indices),
        "candidate_policy": "Filtered head/tail ranking within node type and entity model; all other known triples are filtered.",
        "metrics_definition": "Mean reciprocal rank and Hits@K over both head and tail predictions for held-out observed edges.",
        "graph_training": graph_training, "summaries": summaries, "rows": rows,
        "generation": False, "reranking": False,
        "notes": ("This is held-out link prediction, not question-based retrieval. Graph edges are split by relation; "
                  "GCN message passing and all graph-model fitting use training edges only. Training skips edge/epoch "
                  "pairs for which every same-type corruption is already a known training edge; the skip count is "
                  "reported per method. Node text features omit the graph links being predicted. Test fitting uses "
                  "train plus development edges. Only explicit "
                  "entity relationships, manifest structure and resolved mappings, trait applications and "
                  "inheritance layers in the pinned local CDM projection are included; unresolved manifest "
                  "references are omitted."),
    }
    result["log_persisted"] = bool(persist(result))
    emit({"type": "link_prediction_benchmark_completed", "result": result})
    return result
