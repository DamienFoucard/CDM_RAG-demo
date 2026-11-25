"""Low-memory relationship retrieval evaluation with text and graph embeddings."""
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
from rank_bm25 import BM25Okapi

from app.schema.corpus import load_catalog
from app.schema.graph import SchemaGraph
from app.retrieval.index import tokens
from app.runtime.ollama import ModelError, Ollama

GRAPH_METHODS = ("text_hybrid", "gcn_1", "gcn_2", "gcn_3", "transe", "transh", "boxe")
GRAPH_METHOD_LABELS = {
    "text_hybrid": "Text hybrid",
    "gcn_1": "Trainable GCN · 1 layer",
    "gcn_2": "Trainable GCN · 2 layers",
    "gcn_3": "Trainable GCN · 3 layers",
    "transe": "TransE",
    "transh": "TransH",
    "boxe": "BoxE",
}
GRAPH_DATASET_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "relationship-questions-v1.json"
GRAPH_CHALLENGE_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "relationship-challenge-v1.json"
SEED = 20261122
DIMENSIONS = 32
EPOCHS = 80


def load_relationship_datasets() -> dict[str, Any]:
    main = json.loads(GRAPH_DATASET_PATH.read_text(encoding="utf-8"))
    challenge = json.loads(GRAPH_CHALLENGE_PATH.read_text(encoding="utf-8"))
    if (not isinstance(main, dict) or not isinstance(main.get("cases"), list)
            or not isinstance(challenge, dict) or not isinstance(challenge.get("cases"), list)
            or main.get("source_commit") != challenge.get("source_commit")):
        raise ValueError("The relationship retrieval datasets are invalid or target different CDM revisions.")
    ids: set[str] = set()
    cases = []
    for source, split_name in ((main, None), (challenge, "challenge")):
        for case in source["cases"]:
            split = split_name or case.get("split")
            edges = case.get("gold_edges")
            if (not isinstance(case.get("id"), str) or case["id"] in ids
                    or not isinstance(case.get("question"), str)
                    or split not in {"development", "test", "challenge"}
                    or not isinstance(edges, list) or not edges
                    or any(not isinstance(edge, dict) or not isinstance(edge.get("id"), str)
                           for edge in edges)):
                raise ValueError("A relationship retrieval question is invalid.")
            ids.add(case["id"])
            cases.append({**case, "split": split})
    return {
        "dataset_id": main.get("dataset_id"),
        "challenge_dataset_id": challenge.get("dataset_id"),
        "source_commit": main.get("source_commit"),
        "status": main.get("status"),
        "provenance": main.get("provenance"),
        "challenge_provenance": challenge.get("provenance"),
        "cases": cases,
    }


def _normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12)


def _rank(values: np.ndarray, ids: list[str], *, descending: bool = True) -> list[int]:
    return sorted(range(len(ids)), key=lambda i: ((-float(values[i]) if descending else float(values[i])), ids[i]))


def _rank_positions(ranked: list[int]) -> dict[int, int]:
    return {index: rank for rank, index in enumerate(ranked, 1)}


def _valid_negative(triple: np.ndarray, side: int, known: set[tuple[int, int, int]],
                    pools: list[np.ndarray], rng: np.random.Generator,
                    cache: dict[tuple[int, int, int, int], np.ndarray] | None = None
                    ) -> np.ndarray | None:
    entity_slot = 0 if side == 0 else 2
    source = tuple(int(value) for value in triple)
    cache_key = (*source, side)
    candidates = cache.get(cache_key) if cache is not None else None
    if candidates is None:
        pool = pools[int(triple[entity_slot])]
        valid = []
        for node_index in pool:
            candidate = triple.copy()
            candidate[side] = int(node_index)
            if tuple(int(value) for value in candidate) not in known:
                valid.append(int(node_index))
        candidates = np.asarray(valid, dtype=np.int64)
        if cache is not None:
            cache[cache_key] = candidates
    if not len(candidates):
        return None
    negative = triple.copy()
    negative[side] = int(rng.choice(candidates))
    return negative


def _graph_data(graph: SchemaGraph) -> tuple[list[str], list[str], np.ndarray, list[np.ndarray], list[dict]]:
    entity_ids = sorted(graph.entities)
    relation_names = sorted({edge["attribute"] for edge in graph.edges.values()})
    entity_idx = {value: index for index, value in enumerate(entity_ids)}
    relation_idx = {value: index for index, value in enumerate(relation_names)}
    edges = sorted(graph.edges.values(), key=lambda item: item["id"])
    triples = np.asarray([(entity_idx[edge["source_id"]], relation_idx[edge["attribute"]],
                          entity_idx[edge["target_id"]]) for edge in edges], dtype=np.int64)
    pools_by_model: dict[str, np.ndarray] = {}
    for model in {graph.entities[entity_id]["model"] for entity_id in entity_ids}:
        pools_by_model[model] = np.asarray([entity_idx[entity_id] for entity_id in entity_ids
                                             if graph.entities[entity_id]["model"] == model], dtype=np.int64)
    pools = [pools_by_model[graph.entities[entity_id]["model"]] for entity_id in entity_ids]
    known = {tuple(int(value) for value in triple) for triple in triples}
    return entity_ids, relation_names, triples, pools, edges


def _train_transe(triples: np.ndarray, entity_count: int, relation_count: int,
                  pools: list[np.ndarray], known: set[tuple[int, int, int]],
                  epochs: int = EPOCHS, dimensions: int = DIMENSIONS,
                  return_state: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    entities = rng.normal(0, 0.1, (entity_count, dimensions)).astype(np.float32)
    relations = rng.normal(0, 0.1, (relation_count, dimensions)).astype(np.float32)
    lr, margin = 0.015, 1.0
    negative_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
    negative_sample_skips = 0
    for _ in range(epochs):
        for row in rng.permutation(len(triples)):
            positive = triples[row]
            # Alternating head/tail corruptions avoids relation-ID corruption (side 1).
            side = int(rng.integers(0, 2)) * 2
            negative = _valid_negative(positive, side, known, pools, rng, negative_cache)
            if negative is None:
                negative_sample_skips += 1
                continue
            hp, rp, tp = entities[positive[0]], relations[positive[1]], entities[positive[2]]
            hn, rn, tn = entities[negative[0]], relations[negative[1]], entities[negative[2]]
            qp, qn = hp + rp - tp, hn + rn - tn
            dp, dn = float(np.abs(qp).sum()), float(np.abs(qn).sum())
            if margin + dp - dn <= 0:
                continue
            gp, gn = np.sign(qp), -np.sign(qn)
            entities[positive[0]] -= lr * gp
            relations[positive[1]] -= lr * gp
            entities[positive[2]] += lr * gp
            entities[negative[0]] -= lr * gn
            relations[negative[1]] -= lr * gn
            entities[negative[2]] += lr * gn
        entities[:] = _normalize(entities)
    energies = np.asarray([np.abs(entities[h] + relations[r] - entities[t]).sum()
                           for h, r, t in triples], dtype=np.float32)
    build_seconds = round(time.perf_counter() - start, 6)
    info = {"dimensions": dimensions, "epochs": epochs, "negatives_per_positive": 1,
            "negative_sample_skips": negative_sample_skips,
            "build_seconds": build_seconds, "training_seconds": build_seconds,
            "weights_sha256": hashlib.sha256(entities.tobytes() + relations.tobytes()).hexdigest()}
    if return_state:
        info["_model_state"] = {"entities": entities, "relations": relations}
    return -energies, info


def _transh_score(triple: np.ndarray, entities: np.ndarray, translations: np.ndarray,
                  normals: np.ndarray) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    h, r, t = (int(value) for value in triple)
    normal = normals[r]
    delta = entities[h] - entities[t]
    projection = float(delta @ normal)
    residual = delta - projection * normal + translations[r]
    return float(np.abs(residual).sum()), (normal, delta, residual, projection)


def _train_transh(triples: np.ndarray, entity_count: int, relation_count: int,
                  pools: list[np.ndarray], known: set[tuple[int, int, int]],
                  epochs: int = EPOCHS, dimensions: int = DIMENSIONS,
                  return_state: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    rng = np.random.default_rng(SEED + 1)
    entities = _normalize(rng.normal(0, 0.1, (entity_count, dimensions)).astype(np.float32))
    translations = rng.normal(0, 0.1, (relation_count, dimensions)).astype(np.float32)
    normals = _normalize(rng.normal(0, 0.1, (relation_count, dimensions)).astype(np.float32))
    lr, margin = 0.008, 1.0
    negative_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
    negative_sample_skips = 0
    for _ in range(epochs):
        for row in rng.permutation(len(triples)):
            positive = triples[row]
            side = int(rng.integers(0, 2)) * 2
            negative = _valid_negative(positive, side, known, pools, rng, negative_cache)
            if negative is None:
                negative_sample_skips += 1
                continue
            pos_score, (wp, dp, qp, sp) = _transh_score(positive, entities, translations, normals)
            neg_score, (wn, dn, qn, sn) = _transh_score(negative, entities, translations, normals)
            if margin + pos_score - neg_score <= 0:
                continue
            for triple, normal, delta, residual, projection, coefficient in (
                    (positive, wp, dp, qp, sp, 1.0), (negative, wn, dn, qn, sn, -1.0)):
                h, r, t = (int(value) for value in triple)
                g = coefficient * np.sign(residual)
                projected_g = g - float(g @ normal) * normal
                grad_normal = -projection * g - float(normal @ g) * delta
                entities[h] -= lr * projected_g
                entities[t] += lr * projected_g
                translations[r] -= lr * g
                normals[r] -= lr * grad_normal
                normals[r] /= max(float(np.linalg.norm(normals[r])), 1e-12)
                translations[r] -= normals[r] * float(translations[r] @ normals[r])
        entities[:] = _normalize(entities)
    energies = np.asarray([_transh_score(triple, entities, translations, normals)[0]
                           for triple in triples], dtype=np.float32)
    build_seconds = round(time.perf_counter() - start, 6)
    info = {"dimensions": dimensions, "epochs": epochs, "negatives_per_positive": 1,
            "negative_sample_skips": negative_sample_skips,
            "build_seconds": build_seconds, "training_seconds": build_seconds,
            "weights_sha256": hashlib.sha256(entities.tobytes() + translations.tobytes()
                                               + normals.tobytes()).hexdigest()}
    if return_state:
        info["_model_state"] = {"entities": entities, "translations": translations, "normals": normals}
    return -energies, info


def _box_distance(point: np.ndarray, center: np.ndarray, width: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    delta = point - center
    absolute = np.abs(delta)
    inside = absolute <= ((width - 1.0) / 2.0)
    kappa = 0.5 * (width - 1.0) * (width - 1.0 / width)
    distance = np.where(inside, absolute / width, absolute * width - kappa)
    grad_point = np.where(inside, np.sign(delta) / width, np.sign(delta) * width)
    grad_width = np.where(inside, -absolute / np.square(width),
                          absolute - 0.5 * (2.0 * width - 1.0 - 1.0 / np.square(width)))
    return float(distance.sum()), grad_point, -grad_point, grad_width


def _boxe_score(triple: np.ndarray, entities: np.ndarray, bumps: np.ndarray,
                centers: np.ndarray, widths: np.ndarray) -> tuple[float, tuple]:
    h, r, t = (int(value) for value in triple)
    head_point, tail_point = entities[h] + bumps[t], entities[t] + bumps[h]
    head_score, head_g, head_c, head_w = _box_distance(head_point, centers[r, 0], widths[r, 0])
    tail_score, tail_g, tail_c, tail_w = _box_distance(tail_point, centers[r, 1], widths[r, 1])
    return head_score + tail_score, (head_g, head_c, head_w, tail_g, tail_c, tail_w)


def _train_boxe(triples: np.ndarray, entity_count: int, relation_count: int,
                pools: list[np.ndarray], known: set[tuple[int, int, int]],
                epochs: int = EPOCHS, dimensions: int = DIMENSIONS,
                return_state: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    rng = np.random.default_rng(SEED + 2)
    entities = rng.normal(0, 0.1, (entity_count, dimensions)).astype(np.float32)
    bumps = rng.normal(0, 0.025, (entity_count, dimensions)).astype(np.float32)
    centers = rng.normal(0, 0.1, (relation_count, 2, dimensions)).astype(np.float32)
    widths = np.full((relation_count, 2, dimensions), 1.5, dtype=np.float32)
    lr, margin = 0.004, 1.0
    negative_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
    negative_sample_skips = 0
    for _ in range(epochs):
        for row in rng.permutation(len(triples)):
            positive = triples[row]
            side = int(rng.integers(0, 2)) * 2
            negative = _valid_negative(positive, side, known, pools, rng, negative_cache)
            if negative is None:
                negative_sample_skips += 1
                continue
            pos_score, pos_grad = _boxe_score(positive, entities, bumps, centers, widths)
            neg_score, neg_grad = _boxe_score(negative, entities, bumps, centers, widths)
            if margin + pos_score - neg_score <= 0:
                continue
            for triple, gradients, coefficient in ((positive, pos_grad, 1.0), (negative, neg_grad, -1.0)):
                h, r, t = (int(value) for value in triple)
                gh, gc, gw, gt, gtc, gtw = gradients
                gh, gc, gw, gt, gtc, gtw = (coefficient * value for value in (gh, gc, gw, gt, gtc, gtw))
                entities[h] -= lr * gh
                entities[t] -= lr * gt
                bumps[t] -= lr * gh
                bumps[h] -= lr * gt
                centers[r, 0] -= lr * gc
                centers[r, 1] -= lr * gtc
                widths[r, 0] = np.clip(widths[r, 0] - lr * gw, 1.05, 8.0)
                widths[r, 1] = np.clip(widths[r, 1] - lr * gtw, 1.05, 8.0)
        entities[:] = _normalize(entities)
    energies = np.asarray([_boxe_score(triple, entities, bumps, centers, widths)[0]
                           for triple in triples], dtype=np.float32)
    state = entities.tobytes() + bumps.tobytes() + centers.tobytes() + widths.tobytes()
    build_seconds = round(time.perf_counter() - start, 6)
    info = {"dimensions": dimensions, "epochs": epochs, "negatives_per_positive": 1,
            "negative_sample_skips": negative_sample_skips,
            "build_seconds": build_seconds, "training_seconds": build_seconds,
            "weights_sha256": hashlib.sha256(state).hexdigest()}
    if return_state:
        info["_model_state"] = {"entities": entities, "bumps": bumps,
                                "centers": centers, "widths": widths}
    return -energies, info


def _edge_texts(graph: SchemaGraph, edges: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    edge_docs, node_docs, relation_docs = [], [], []
    for edge in edges:
        source, target = graph.entities[edge["source_id"]], graph.entities[edge["target_id"]]
        edge_docs.append(f"{source['name']} {source.get('description', '')} -- {edge['attribute']} "
                         f"{edge.get('description', '')} --> {target['name']} {target.get('description', '')}")
    for entity_id in sorted(graph.entities):
        entity = graph.entities[entity_id]
        attributes = ", ".join(item["name"] for item in entity.get("attributes", [])[:20])
        node_docs.append(f"{entity['name']}. {entity.get('description', '')}. Attributes: {attributes}")
    relation_names = sorted({edge["attribute"] for edge in edges})
    for name in relation_names:
        descriptions = sorted({edge.get("description", "") for edge in edges
                               if edge["attribute"] == name and edge.get("description")})
        relation_docs.append(name + (". " + " ".join(descriptions[:2]) if descriptions else ""))
    return edge_docs, node_docs, relation_docs


def _historical_embedding_build_seconds(data_dir: Path, identity: str) -> float | None:
    """Recover the original cold index-build duration for pre-metadata caches."""
    log_path = data_dir / "retrieval_benchmark_runs.jsonl"
    if not log_path.exists():
        return None
    try:
        with log_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or record.get("track") != "relationships":
                    continue
                summaries = record.get("summaries")
                if not isinstance(summaries, list):
                    continue
                for summary in summaries:
                    if not isinstance(summary, dict) or summary.get("index_identity") != identity:
                        continue
                    seconds = summary.get("index_build_seconds")
                    if not isinstance(seconds, (int, float)) and not summary.get("index_reused"):
                        seconds = summary.get("index_preparation_seconds")
                    if isinstance(seconds, (int, float)) and seconds >= 0:
                        return float(seconds)
    except OSError:
        return None
    return None


def _load_or_build_embeddings(model: Ollama, graph: SchemaGraph, edges: list[dict[str, Any]],
                              data_dir: Path, emit: Callable[[dict[str, Any]], None],
                              model_name: str, model_position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    edge_docs, node_docs, relation_docs = _edge_texts(graph, edges)
    digest = model.embedding_digest()
    identity = hashlib.sha256(json.dumps({"revision": graph.revision, "projection": graph.projection_revision,
                                          "model": model.embedding_model, "digest": digest,
                                          "format": model.embedding_format(), "edge_docs": edge_docs,
                                          "node_docs": node_docs, "relation_docs": relation_docs},
                                         sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache_dir = data_dir / "retrieval_benchmark_indices" / "relationship_embeddings"
    cache_path = cache_dir / f"{identity}.npz"
    start = time.perf_counter()
    if cache_path.exists():
        index_build_seconds = None
        try:
            with np.load(cache_path, allow_pickle=False) as saved:
                edge_vectors = saved["edge_vectors"].copy()
                node_vectors = saved["node_vectors"].copy()
                relation_vectors = saved["relation_vectors"].copy()
                if "build_seconds" in saved.files:
                    index_build_seconds = float(np.asarray(saved["build_seconds"]).item())
            cache_valid = (edge_vectors.shape[0] == len(edges)
                           and node_vectors.shape[0] == len(graph.entities)
                           and relation_vectors.shape[0] == len({edge["attribute"] for edge in edges}))
        except (OSError, ValueError, KeyError, EOFError):
            cache_valid = False
        if cache_valid:
            if index_build_seconds is None:
                index_build_seconds = _historical_embedding_build_seconds(data_dir, identity)
            return edge_vectors, node_vectors, relation_vectors, {
                "index_identity": identity, "index_preparation_seconds": round(time.perf_counter() - start, 3),
                "index_build_seconds": index_build_seconds,
                "index_reused": True, "model_digest": digest, "embedding_format": model.embedding_format()}
        cache_path.unlink(missing_ok=True)
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    build_started = time.perf_counter()
    all_docs = edge_docs + node_docs + relation_docs
    batches = []
    for offset in range(0, len(all_docs), 24):
        emit({"type": "relationship_embedding_batch", "model": model_name,
              "position": model_position, "completed": min(offset + 24, len(all_docs)),
              "total": len(all_docs)})
        batches.extend(model.embed(all_docs[offset:offset + 24], query=False, keep_alive="5m"))
    vectors = _normalize(np.asarray(batches, dtype=np.float32))
    e_end = len(edge_docs)
    n_end = e_end + len(node_docs)
    edge_vectors, node_vectors, relation_vectors = vectors[:e_end], vectors[e_end:n_end], vectors[n_end:]
    temporary = cache_path.with_suffix(".npz.tmp")
    index_build_seconds = round(time.perf_counter() - build_started, 3)
    with temporary.open("wb") as stream:
        np.savez(stream, edge_vectors=edge_vectors, node_vectors=node_vectors,
                 relation_vectors=relation_vectors,
                 build_seconds=np.asarray(index_build_seconds, dtype=np.float64))
    temporary.replace(cache_path)
    return edge_vectors, node_vectors, relation_vectors, {
        "index_identity": identity, "index_preparation_seconds": round(time.perf_counter() - start, 3),
        "index_build_seconds": index_build_seconds,
        "index_reused": False, "model_digest": digest, "embedding_format": model.embedding_format()}


def _aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    values = [row["variants"][method] for row in rows]
    latencies = sorted(float(value["latency_ms"]) for value in values)
    if not values:
        return {"cases": 0}
    p95 = latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)]
    return {
        "cases": len(values),
        "edge_recall_at_20": round(statistics.mean(float(item["edge_recall_at_20"]) for item in values), 4),
        "all_edges_at_20": round(statistics.mean(float(item["all_edges_at_20"]) for item in values), 4),
        "first_edge_mrr_at_20": round(statistics.mean(float(item["first_edge_mrr_at_20"]) for item in values), 4),
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "p95_latency_ms": round(p95, 2),
    }


def _normalized_graph_edges(edges: list[dict[str, Any]], entity_to_index: dict[str, int]
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a sparse, symmetric normalized adjacency with self loops."""
    node_count = len(entity_to_index)
    arcs: set[tuple[int, int]] = {(index, index) for index in range(node_count)}
    for edge in edges:
        source, target = entity_to_index[edge["source_id"]], entity_to_index[edge["target_id"]]
        arcs.add((source, target))
        arcs.add((target, source))
    ordered = sorted(arcs)
    sources = np.asarray([source for source, _ in ordered], dtype=np.int64)
    targets = np.asarray([target for _, target in ordered], dtype=np.int64)
    degree = np.bincount(targets, minlength=node_count).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(degree[sources] * degree[targets], 1.0))
    return sources, targets, weights.astype(np.float32)


def _aggregate_graph(features: np.ndarray, sources: np.ndarray, targets: np.ndarray,
                     weights: np.ndarray) -> np.ndarray:
    """Apply sparse normalized adjacency without allocating a dense node matrix."""
    aggregated = np.zeros_like(features)
    np.add.at(aggregated, targets, features[sources] * weights[:, None])
    return aggregated


def _train_gcn(method: str, node_features: np.ndarray, triples: np.ndarray,
               relation_count: int, pools: list[np.ndarray], known: set[tuple[int, int, int]],
               edges: list[dict[str, Any]], entity_to_index: dict[str, int],
               epochs: int = EPOCHS, dimensions: int = DIMENSIONS,
               return_state: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a feature-based GCN encoder for relational edge prediction."""
    if method not in {"gcn_1", "gcn_2", "gcn_3"}:
        raise ValueError("A trainable GCN depth must be selected.")
    start = time.perf_counter()
    depth = int(method[-1])
    node_count = node_features.shape[0]
    if node_count != len(entity_to_index) or len(triples) != len(edges):
        raise ValueError("GCN input features, entities, and graph edges do not align.")

    sources, targets, adjacency_weights = _normalized_graph_edges(edges, entity_to_index)
    rng = np.random.default_rng(SEED + 10 + depth)
    weights: list[np.ndarray] = []
    input_size = node_features.shape[1]
    for layer in range(depth):
        fan_in = input_size if layer == 0 else dimensions
        bound = math.sqrt(6.0 / (fan_in + dimensions))
        weights.append(rng.uniform(-bound, bound, (fan_in, dimensions)).astype(np.float32))
    relations = rng.normal(0, 0.1, (relation_count, dimensions)).astype(np.float32)
    parameters = [*weights, relations]
    first_moments = [np.zeros_like(value) for value in parameters]
    second_moments = [np.zeros_like(value) for value in parameters]
    beta1, beta2, learning_rate, weight_decay = 0.9, 0.999, 0.003, 1e-5
    negative_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
    negative_sample_skips = 0

    for epoch in range(1, epochs + 1):
        negative_triples = []
        negative_mask = []
        for triple in triples:
            side = int(rng.integers(0, 2)) * 2
            negative = _valid_negative(triple, side, known, pools, rng, negative_cache)
            if negative is None:
                negative_sample_skips += 1
                negative_triples.append(triple.copy())
                negative_mask.append(0.0)
            else:
                negative_triples.append(negative)
                negative_mask.append(1.0)
        negatives = np.asarray(negative_triples, dtype=np.int64)

        layer_inputs: list[np.ndarray] = []
        preactivations: list[np.ndarray] = []
        hidden = node_features
        for layer, layer_weights in enumerate(weights):
            aggregated = _aggregate_graph(hidden, sources, targets, adjacency_weights)
            preactivation = aggregated @ layer_weights
            layer_inputs.append(aggregated)
            preactivations.append(preactivation)
            hidden = np.maximum(preactivation, 0.0) if layer < depth - 1 else preactivation
        node_embeddings = hidden

        heads, relation_ids, tails = triples.T
        neg_heads, neg_relations, neg_tails = negatives.T
        positive_logits = np.sum(node_embeddings[heads] * relations[relation_ids]
                                 * node_embeddings[tails], axis=1)
        negative_logits = np.sum(node_embeddings[neg_heads] * relations[neg_relations]
                                 * node_embeddings[neg_tails], axis=1)
        positive_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(positive_logits, -40.0, 40.0)))
        negative_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(negative_logits, -40.0, 40.0)))
        scale = 0.5 / max(len(triples), 1)
        sample_mask = np.asarray(negative_mask, dtype=np.float32)
        positive_gradient = (positive_sigmoid - 1.0) * scale * sample_mask
        negative_gradient = negative_sigmoid * scale * sample_mask

        node_gradient = np.zeros_like(node_embeddings)
        relation_gradient = np.zeros_like(relations)
        for h, r, t, gradient in zip(heads, relation_ids, tails, positive_gradient):
            node_gradient[h] += gradient * relations[r] * node_embeddings[t]
            node_gradient[t] += gradient * relations[r] * node_embeddings[h]
            relation_gradient[r] += gradient * node_embeddings[h] * node_embeddings[t]
        for h, r, t, gradient in zip(neg_heads, neg_relations, neg_tails, negative_gradient):
            node_gradient[h] += gradient * relations[r] * node_embeddings[t]
            node_gradient[t] += gradient * relations[r] * node_embeddings[h]
            relation_gradient[r] += gradient * node_embeddings[h] * node_embeddings[t]

        weight_gradients = [np.zeros_like(value) for value in weights]
        gradient = node_gradient
        for layer in range(depth - 1, -1, -1):
            if layer < depth - 1:
                gradient = gradient * (preactivations[layer] > 0.0)
            weight_gradients[layer] = layer_inputs[layer].T @ gradient
            propagated_gradient = gradient @ weights[layer].T
            gradient = _aggregate_graph(propagated_gradient, targets, sources, adjacency_weights)
        gradients = [*weight_gradients, relation_gradient]
        for index, (parameter, parameter_gradient) in enumerate(zip(parameters, gradients)):
            parameter_gradient += weight_decay * parameter
            first_moments[index] *= beta1
            first_moments[index] += (1.0 - beta1) * parameter_gradient
            second_moments[index] *= beta2
            second_moments[index] += (1.0 - beta2) * np.square(parameter_gradient)
            corrected_first = first_moments[index] / (1.0 - beta1 ** epoch)
            corrected_second = second_moments[index] / (1.0 - beta2 ** epoch)
            parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)

    hidden = node_features
    for layer, layer_weights in enumerate(weights):
        aggregated = _aggregate_graph(hidden, sources, targets, adjacency_weights)
        preactivation = aggregated @ layer_weights
        hidden = np.maximum(preactivation, 0.0) if layer < depth - 1 else preactivation
    heads, relation_ids, tails = triples.T
    scores = np.sum(hidden[heads] * relations[relation_ids] * hidden[tails], axis=1).astype(np.float32)
    build_seconds = round(time.perf_counter() - start, 6)
    state = b"".join(value.tobytes() for value in parameters)
    info = {
        "training_kind": "trainable_gcn",
        "dimensions": dimensions, "epochs": epochs, "layers": depth,
        "negatives_per_positive": 1,
        "negative_sample_skips": negative_sample_skips,
        "objective": "DistMult binary cross-entropy with one filtered negative when available; saturated typed candidates are skipped",
        "build_seconds": build_seconds, "training_seconds": build_seconds,
        "weights_sha256": hashlib.sha256(state).hexdigest(),
    }
    if return_state:
        info["_model_state"] = {"node_embeddings": hidden, "relations": relations}
    return scores, info


def run_relationship_retrieval_benchmark(catalog_path: Path, data_dir: Path, configured_models: Ollama,
                                         model_names: list[str], question_ids: list[str], split: str,
                                         graph_methods: list[str], dataset: dict[str, Any], run_id: str, model_lock,
                                         emit: Callable[[dict[str, Any]], None], persist) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if split not in {"development", "test", "challenge"}:
        raise ValueError("Choose development, test, or challenge relationship questions.")
    if (not model_names or len(model_names) > 6 or len(set(model_names)) != len(model_names)
            or not question_ids or len(question_ids) > 42 or len(set(question_ids)) != len(question_ids)):
        raise ValueError("Select one to six distinct embedding models and between one and 42 questions.")
    from app.evaluation.retrieval_benchmark import available_embedding_models
    installed = {item["name"] for item in available_embedding_models(configured_models.base_url)}
    if not set(model_names).issubset(installed):
        raise ValueError("Select only installed embedding models with supported query formats.")
    cases_by_id = {item["id"]: item for item in dataset["cases"]}
    if any(item not in cases_by_id or cases_by_id[item]["split"] != split for item in question_ids):
        raise ValueError("Select questions from the chosen relationship evaluation split.")
    cases = [cases_by_id[item] for item in question_ids]
    methods = graph_methods
    if not methods or len(set(methods)) != len(methods) or not set(methods).issubset(GRAPH_METHODS):
        raise ValueError("Select one or more supported graph retrieval methods.")
    catalog = load_catalog(catalog_path)
    graph = SchemaGraph(catalog["entities"], catalog["source_commit"])
    if dataset["source_commit"] != graph.revision:
        raise ValueError("The relationship questions and local catalog target different CDM revisions.")
    edges = sorted(graph.edges.values(), key=lambda item: item["id"])
    edge_index = {edge["id"]: index for index, edge in enumerate(edges)}
    missing = {gold["id"] for case in cases for gold in case["gold_edges"]} - set(edge_index)
    if missing:
        raise ValueError("A labeled relationship is missing from this catalog revision.")
    entity_ids, relation_names, triples, pools, _ = _graph_data(graph)
    entity_to_index = {value: index for index, value in enumerate(entity_ids)}
    known = {tuple(int(value) for value in triple) for triple in triples}

    # KGE embeddings depend only on the catalog, so fit them once per run and share
    # their edge scores across selected text embedding models. GCN fitting uses the
    # selected model's node features and therefore happens once per model below.
    graph_models: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for method, trainer in (("transe", _train_transe), ("transh", _train_transh), ("boxe", _train_boxe)):
        if method in methods:
            emit({"type": "relationship_graph_training_started", "run_id": run_id,
                  "method": method, "candidate_edge_count": len(edges),
                  "message": f"Training {GRAPH_METHOD_LABELS[method]} on observed CDM edges."})
            scores, info = trainer(triples, len(entity_ids), len(relation_names), pools, known)
            graph_models[method] = (scores, info)
            emit({"type": "relationship_graph_training_completed", "run_id": run_id,
                  "method": method, "candidate_edge_count": len(edges),
                  "build_seconds": info["build_seconds"], "training_seconds": info["training_seconds"],
                  "training_info": info})

    summaries, rows = [], []
    with model_lock:
        for model_position, model_name in enumerate(model_names, 1):
            emit({"type": "relationship_model_started", "run_id": run_id, "model": model_name,
                  "position": model_position, "total": len(model_names)})
            model = Ollama(configured_models.base_url, model_name,
                           configured_models.router_model, configured_models.answer_model,
                           configured_models.scope_model, configured_models.judge_model,
                           configured_models.request_timeout, configured_models.stream_idle_timeout,
                           configured_models.first_token_timeout)
            model.query_embedding_keep_alive = "5m"
            try:
                edge_vectors, node_vectors, relation_vectors, index_info = _load_or_build_embeddings(
                    model, graph, edges, data_dir, emit, model_name, model_position)
                emit({"type": "relationship_embedding_warmup_started", "run_id": run_id,
                      "model": model_name, "question_id": cases[0]["id"]})
                warmup_started = time.perf_counter()
                model.embed([cases[0]["question"]], query=True, keep_alive="5m")
                query_warmup_ms = round((time.perf_counter() - warmup_started) * 1000, 2)
                emit({"type": "relationship_embedding_warmup_completed", "run_id": run_id,
                      "model": model_name, "warmup_ms": query_warmup_ms})
                documents = [f"{edge['source_id']} {edge['attribute']} {edge['target_id']} "
                             f"{edge.get('description', '')}" for edge in edges]
                bm25 = BM25Okapi([tokens(document) for document in documents])
                normalized_edges = _normalize(edge_vectors)
                method_edge_scores = {method: scores for method, (scores, _) in graph_models.items()}
                graph_method_build_seconds = {}
                graph_method_build_info = {}
                for method in methods:
                    if not method.startswith("gcn_"):
                        continue
                    emit({"type": "relationship_graph_training_started", "run_id": run_id,
                          "method": method, "model": model_name, "candidate_edge_count": len(edges),
                          "message": f"Training {GRAPH_METHOD_LABELS[method]} with {model_name} text features."})
                    method_edge_scores[method], info = _train_gcn(
                        method, node_vectors, triples, len(relation_names), pools, known,
                        edges, entity_to_index)
                    graph_method_build_seconds[method] = info["build_seconds"]
                    graph_method_build_info[method] = info
                    emit({"type": "relationship_graph_training_completed", "run_id": run_id,
                          "method": method, "model": model_name, "candidate_edge_count": len(edges),
                          "build_seconds": info["build_seconds"],
                          "training_seconds": info["training_seconds"], "training_info": info})
                model_rows: list[dict[str, Any]] = []
                for question_position, case in enumerate(cases, 1):
                    emit({"type": "relationship_question_started", "run_id": run_id, "model": model_name,
                          "question_id": case["id"], "question": case["question"],
                          "position": question_position, "total": len(cases)})
                    search_started = time.perf_counter()
                    query = _normalize(np.asarray(model.embed([case["question"]], query=True,
                                                              keep_alive="5m"), dtype=np.float32))[0]
                    lexical_scores = np.asarray(bm25.get_scores(tokens(case["question"])), dtype=np.float32)
                    semantic_scores = normalized_edges @ query
                    shared_elapsed_ms = (time.perf_counter() - search_started) * 1000
                    lexical_order = _rank(lexical_scores, [edge["id"] for edge in edges])
                    semantic_order = _rank(semantic_scores, [edge["id"] for edge in edges])
                    lexical_rank, semantic_rank = _rank_positions(lexical_order), _rank_positions(semantic_order)
                    method_latencies: dict[str, float] = {}
                    method_started = time.perf_counter()
                    fused: dict[str, list[int]] = {"text_hybrid": _rank(
                        np.asarray([1 / (60 + lexical_rank[i]) + 1 / (60 + semantic_rank[i]) for i in range(len(edges))]),
                        [edge["id"] for edge in edges])}
                    method_latencies["text_hybrid"] = round(shared_elapsed_ms +
                                                              (time.perf_counter() - method_started) * 1000, 2)
                    for method in methods:
                        if method == "text_hybrid":
                            continue
                        method_started = time.perf_counter()
                        graph_order = _rank(method_edge_scores[method], [edge["id"] for edge in edges])
                        graph_rank = _rank_positions(graph_order)
                        scores = np.asarray([1 / (60 + lexical_rank[i]) + 1 / (60 + semantic_rank[i])
                                             + 0.5 / (60 + graph_rank[i]) for i in range(len(edges))])
                        fused[method] = _rank(scores, [edge["id"] for edge in edges])
                        method_latencies[method] = round(shared_elapsed_ms +
                                                         (time.perf_counter() - method_started) * 1000, 2)
                    variants = {}
                    gold = {edge["id"] for edge in case["gold_edges"]}
                    for method in methods:
                        ranked_ids = [edges[index]["id"] for index in fused[method][:20]]
                        matched = gold.intersection(ranked_ids)
                        first = next((rank for rank, value in enumerate(ranked_ids, 1) if value in gold), None)
                        variants[method] = {
                            "ranked_edge_ids": ranked_ids,
                            "matched_edge_ids": [value for value in ranked_ids if value in gold],
                            "edge_recall_at_20": len(matched) / len(gold),
                            "all_edges_at_20": gold.issubset(ranked_ids),
                            "first_edge_mrr_at_20": 1 / first if first else 0.0,
                            "latency_ms": method_latencies[method],
                        }
                    row = {"model": model_name, "question_id": case["id"], "question": case["question"],
                           "split": case["split"], "type": case.get("type"),
                           "gold_edge_ids": sorted(gold), "variants": variants}
                    rows.append(row)
                    model_rows.append(row)
                    emit({"type": "relationship_row", "run_id": run_id, "row": row})
                summary = {
                    "model": model_name, "question_count": len(model_rows),
                    "query_warmup_ms": query_warmup_ms, **index_info,
                    "runtime": model.runtime_info(model_name),
                    "graph_methods": {method: info for method, (_, info) in graph_models.items()},
                    "graph_method_build_seconds": graph_method_build_seconds,
                    "graph_method_build_info": graph_method_build_info,
                    "metrics": {method: _aggregate(model_rows, method) for method in methods},
                }
                summaries.append(summary)
                emit({"type": "relationship_model_completed", "run_id": run_id,
                      "model": model_name, "summary": summary, "position": model_position,
                      "total": len(model_names)})
            finally:
                try:
                    model.unload_if_loaded(model_name)
                    emit({"type": "relationship_model_unloaded", "run_id": run_id, "model": model_name})
                except ModelError as exc:
                    emit({"type": "relationship_unload_warning", "run_id": run_id,
                          "model": model_name, "detail": str(exc)})

    result = {
        "track": "relationships", "run_id": run_id, "started_at": started_at,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "dataset_id": dataset["dataset_id"], "challenge_dataset_id": dataset["challenge_dataset_id"],
        "dataset_status": dataset["status"], "dataset_provenance": dataset["provenance"],
        "challenge_provenance": dataset["challenge_provenance"], "source_commit": dataset["source_commit"],
        "split": split, "model_names": model_names, "question_ids": question_ids,
        "graph_methods": methods, "candidate_edge_count": len(edges), "top_k_edges": 20,
        "reranking": False, "generation": False,
        "graph_training": {method: info for method, (_, info) in graph_models.items()},
        "notes": ("All graph methods rank observed CDM edges; graph embedding methods train without query labels "
                  "on the full observed graph and are not held-out link-prediction evaluations. GCN uses selected "
                  "text-derived entity features, symmetric normalized adjacency with self-loops, trainable graph "
                  "convolution weights, a DistMult decoder, and filtered negative triples for one, two, or three layers. "
                  "GCN fitting is repeated for each selected text embedding model; TransE, TransH, and BoxE fit once per run. Text hybrid fuses "
                  "BM25 and the selected text embedding model. Graph variants add a reciprocal-rank structural "
                  "signal. Questions and labels are synthetic, source-checked, incomplete, and not independently "
                  "reviewed; unlabeled edges may also be relevant. Per-question latency includes query embedding."),
        "summaries": summaries, "rows": rows,
    }
    result["log_persisted"] = persist(result)
    emit({"type": "relationship_benchmark_completed", "result": result})
    return result
