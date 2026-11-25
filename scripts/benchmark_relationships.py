#!/usr/bin/env python3
"""Compare text, neighborhood text, graph traversal and TransE for CDM evidence retrieval."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

try:
    import torch
except ModuleNotFoundError:  # Optional research-only trainer; not needed by the serving app.
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.retrieval.index import SchemaIndex, tokens
from app.runtime.ollama import ModelError, Ollama
from app.retrieval.rag import expand_context, rerank

GENERIC = re.compile(r"created|modified|onbehalf|regardingobject|transactioncurrency", re.I)
VARIANTS = ("hybrid_raw", "hybrid_reranked", "neighborhood_reranked", "graph_assisted", "graph_transe")


def source_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return edge["source_id"], edge["attribute"], edge["target_id"]


def neighborhood_suffix(index: SchemaIndex, entity_id: str, limit: int = 12) -> str:
    """Fixed, query-independent typed edges; prefer outgoing and nonadministrative references."""
    by_id = {edge["id"]: edge for edge in index.graph.adjacency[entity_id]}
    edges = sorted(by_id.values(), key=lambda edge: (
        bool(GENERIC.search(edge["attribute"])), edge["source_id"] != entity_id,
        edge["attribute"], edge["source_id"], edge["target_id"]))[:limit]
    if not edges:
        return " No resolved graph references in this snapshot."
    return " Graph references: " + "; ".join(
        f"{edge['source_id']} --{edge['attribute']}--> {edge['target_id']}" for edge in edges) + "."


class NeighborhoodIndex:
    """The same chunk IDs and hybrid algorithm, with fixed graph context appended to each passage."""

    def __init__(self, base: SchemaIndex, models: Ollama, directory: Path):
        self.models = models
        suffixes = {entity_id: neighborhood_suffix(base, entity_id) for entity_id in base.graph.entities}
        self.chunks = [{**chunk, "text": chunk["text"] + suffixes[chunk["entity_id"]]} for chunk in base.chunks]
        self.by_id = {item["id"]: item for item in self.chunks}
        self.tokenized = [set(tokens(item["text"])) for item in self.chunks]
        self.bm25 = BM25Okapi([tokens(item["text"]) for item in self.chunks])
        identity = hashlib.sha256(json.dumps({"chunks": self.chunks, "digest": base.model_digest,
                                               "format": models.embedding_format()},
                                              sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.identity = identity
        directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(directory))
        name = "rel_neighborhood_" + identity[:20]
        collection = self.client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"},
                                                           embedding_function=None)
        if set(collection.get(include=[]).get("ids", [])) != set(self.by_id):
            self.client.delete_collection(name)
            collection = self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"},
                                                        embedding_function=None)
            for start in range(0, len(self.chunks), 24):
                batch = self.chunks[start:start + 24]
                vectors = models.embed([chunk["text"] for chunk in batch], query=False)
                collection.add(ids=[chunk["id"] for chunk in batch], embeddings=vectors,
                               documents=[chunk["text"] for chunk in batch])
                print(f"Neighborhood embedding: {min(start + len(batch), len(self.chunks))}/{len(self.chunks)}", flush=True)
        self.collection = collection

    def search(self, question: str, limit: int = 20) -> list[dict[str, Any]]:
        words = tokens(question)
        if not words:
            return []
        lexical = [i for i, words_here in enumerate(self.tokenized) if words_here.intersection(words)]
        scores = self.bm25.get_scores(words)
        lexical.sort(key=lambda i: (-float(scores[i]), self.chunks[i]["id"]))
        lexical_ids = [self.chunks[i]["id"] for i in lexical[:limit]]
        query = self.models.embed([question], query=True)[0]
        vector_ids = self.collection.query(query_embeddings=[query], n_results=min(limit, len(self.chunks)))["ids"][0]
        lexical_rank = {key: rank for rank, key in enumerate(lexical_ids, 1)}
        vector_rank = {key: rank for rank, key in enumerate(vector_ids, 1)}
        ids = list(dict.fromkeys(lexical_ids + vector_ids))
        score = lambda key: ((1 / (60 + lexical_rank[key])) if key in lexical_rank else 0) + (
            (1 / (60 + vector_rank[key])) if key in vector_rank else 0)
        ids.sort(key=lambda key: (-score(key), key))
        return [{**self.by_id[key], "score": round(score(key), 6)} for key in ids[:limit]]


def train_transe(graph, dimensions: int = 32, epochs: int = 300, negatives: int = 3,
                 seed: int = 20261122) -> tuple[dict[str, float], dict[str, Any]]:
    """Train on all observed triples; scores rank only observed edges, never assert predictions."""
    if torch is None:
        raise RuntimeError("Install requirements-graph-benchmark.txt to run this offline TransE experiment.")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    entities = sorted(graph.entities)
    relations = sorted({edge["attribute"] for edge in graph.edges.values()})
    entity_id = {value: i for i, value in enumerate(entities)}
    relation_id = {value: i for i, value in enumerate(relations)}
    edges = sorted(graph.edges.values(), key=lambda edge: edge["id"])
    triples = np.array([(entity_id[e["source_id"]], relation_id[e["attribute"]], entity_id[e["target_id"]])
                        for e in edges], dtype=np.int64)
    known = {tuple(row) for row in triples.tolist()}
    # Corrupt a head or tail with another entity in the same CDM model. Only
    # observed triples can be filtered; unobserved triples are not known false.
    pools: dict[str, np.ndarray] = {}
    for model in {graph.entities[e]["model"] for e in entities}:
        pools[model] = np.array([entity_id[e] for e in entities if graph.entities[e]["model"] == model])
    entity_pools = [pools[graph.entities[e]["model"]] for e in entities]
    embedding = torch.nn.Embedding(len(entities), dimensions)
    relation = torch.nn.Embedding(len(relations), dimensions)
    torch.nn.init.xavier_uniform_(embedding.weight)
    torch.nn.init.xavier_uniform_(relation.weight)
    optimizer = torch.optim.Adam([*embedding.parameters(), *relation.parameters()], lr=0.01)
    positive = torch.from_numpy(np.repeat(triples, negatives, axis=0))
    losses = []
    start = time.perf_counter()
    for epoch in range(epochs):
        negative = positive.numpy().copy()
        sides = rng.integers(0, 2, size=len(negative)) * 2
        for i, side in enumerate(sides):
            pool = entity_pools[negative[i, side]]
            for _attempt in range(100):
                candidate = int(rng.choice(pool))
                trial = negative[i].copy()
                trial[side] = candidate
                if tuple(trial) not in known:
                    negative[i] = trial
                    break
            else:
                raise RuntimeError("could not sample an unobserved, same-model negative")
        neg = torch.from_numpy(negative)
        pos_dist = torch.linalg.vector_norm(embedding(positive[:, 0]) + relation(positive[:, 1]) -
                                            embedding(positive[:, 2]), ord=1, dim=1)
        neg_dist = torch.linalg.vector_norm(embedding(neg[:, 0]) + relation(neg[:, 1]) -
                                            embedding(neg[:, 2]), ord=1, dim=1)
        loss = torch.relu(1.0 + pos_dist - neg_dist).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            embedding.weight[:] = torch.nn.functional.normalize(embedding.weight, p=2, dim=1)
        if epoch in {0, epochs - 1}:
            losses.append(float(loss.detach()))
    with torch.no_grad():
        values = torch.from_numpy(triples)
        distances = torch.linalg.vector_norm(embedding(values[:, 0]) + relation(values[:, 1]) -
                                             embedding(values[:, 2]), ord=1, dim=1).numpy()
        state = np.concatenate((embedding.weight.numpy().ravel(), relation.weight.numpy().ravel()))
    return {edge["id"]: -float(distance) for edge, distance in zip(edges, distances)}, {
        "model": "TransE", "dimensions": dimensions, "epochs": epochs, "negatives_per_positive": negatives,
        "seed": seed, "margin": 1.0, "distance": "L1", "learning_rate": 0.01,
        "training_seconds": round(time.perf_counter() - start, 2), "initial_loss": round(losses[0], 6),
        "final_loss": round(losses[-1], 6), "weights_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        "training_scope": "all observed graph edges; unobserved same-model corruptions; not link-prediction evaluation"}


class GraphExplorer:
    """Rank actual edges within two hops of text candidates or explicitly named entities."""

    def __init__(self, index: SchemaIndex, models: Ollama, transe: dict[str, float]):
        self.index, self.models, self.transe = index, models, transe
        self.edges = sorted(index.graph.edges.values(), key=lambda edge: edge["id"])
        self.documents = [f"{e['source_id']} {e['attribute']} {e['target_id']} {e.get('description', '')}" for e in self.edges]
        self.bm25 = BM25Okapi([tokens(text) for text in self.documents])
        chunk_by_pair = {(entity["id"], relation["attribute"]): f"{entity['id']}:relationship:{position}"
                         for entity in index.catalog["entities"]
                         for position, relation in enumerate(entity.get("relationships", []), 1)}
        chunk_ids = list(dict.fromkeys(chunk_by_pair[(e["source_id"], e["attribute"])] for e in self.edges))
        saved = index.collection.get(ids=chunk_ids, include=["embeddings"])
        vectors = np.asarray(saved["embeddings"], dtype=np.float32)
        vector_by_id = {key: vector for key, vector in zip(saved["ids"], vectors)}
        matrix = np.stack([vector_by_id[chunk_by_pair[(e["source_id"], e["attribute"])]] for e in self.edges])
        self.matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    def rank(self, question: str, seed_ids: list[str], *, limit: int = 20) -> tuple[list[str], list[str], dict]:
        question_words = tokens(question)
        seed_order = list(dict.fromkeys(seed_ids))
        # Matching a name in the question is graph entity linking, not a gold label.
        for entity_id, entity in sorted(self.index.graph.entities.items()):
            name_words = tokens(entity["name"])
            if not name_words or len(entity["name"]) < 3:
                continue
            if any(question_words[i:i+len(name_words)] == name_words
                   for i in range(len(question_words) - len(name_words) + 1)) and entity_id not in seed_order:
                seed_order.append(entity_id)
        seed_order = seed_order[:12]
        distances = {entity_id: 0 for entity_id in seed_order}
        queue = deque(seed_order)
        while queue:
            current = queue.popleft()
            if distances[current] >= 1:
                continue
            for edge in self.index.graph.adjacency[current]:
                other = edge["target_id"] if edge["source_id"] == current else edge["source_id"]
                if other not in distances:
                    distances[other] = distances[current] + 1
                    queue.append(other)
        eligible = [i for i, edge in enumerate(self.edges)
                    if edge["source_id"] in distances or edge["target_id"] in distances]
        if not eligible:
            return [], [], {"seed_ids": seed_order, "candidate_edges": 0}
        query_vector = np.asarray(self.models.embed([question], query=True)[0], dtype=np.float32)
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
        semantic = self.matrix @ query_vector
        lexical = self.bm25.get_scores(question_words)
        seed_rank = {entity_id: rank for rank, entity_id in enumerate(seed_order)}

        def proximity(i: int) -> tuple[int, int, str]:
            edge = self.edges[i]
            distance = min(distances.get(edge["source_id"], 99), distances.get(edge["target_id"], 99))
            rank = min(seed_rank.get(edge["source_id"], 999), seed_rank.get(edge["target_id"], 999))
            return distance, rank, edge["id"]

        def positions(key):
            return {value: rank for rank, value in enumerate(sorted(eligible, key=key), 1)}

        lex_rank = positions(lambda i: (-float(lexical[i]), self.edges[i]["id"]))
        sem_rank = positions(lambda i: (-float(semantic[i]), self.edges[i]["id"]))
        prox_rank = positions(proximity)
        transe_rank = positions(lambda i: (-self.transe[self.edges[i]["id"]], self.edges[i]["id"]))

        def base_score(i):
            return 1/(20+lex_rank[i]) + 1/(20+sem_rank[i]) + 0.5/(20+prox_rank[i])

        base = sorted(eligible, key=lambda i: (-base_score(i), self.edges[i]["id"]))
        learned = sorted(eligible, key=lambda i: (-(base_score(i) + 0.5/(20+transe_rank[i])), self.edges[i]["id"]))
        return ([self.edges[i]["id"] for i in base[:limit]],
                [self.edges[i]["id"] for i in learned[:limit]],
                {"seed_ids": seed_order, "candidate_edges": len(eligible)})


def contexts_and_edges(index: SchemaIndex, question: str, candidates: list[dict],
                       selected: list[dict], by_triple: dict) -> tuple[list[str], list[str]]:
    explicit = {}
    for candidate in candidates:
        entity = index.graph.entities[candidate["entity_id"]]
        name = entity["name"]
        if len(name) > 3 and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", question, re.I):
            explicit.setdefault(name.casefold(), candidate)
    contexts, _sources, _omitted = expand_context(index, list(explicit.values()) + selected)
    edge_ids = []
    for context in contexts:
        for edge in context["direct_neighbors"]:
            key = (edge["source_entity_id"], edge["attribute"], edge["target_entity_id"])
            if key in by_triple and by_triple[key] not in edge_ids:
                edge_ids.append(by_triple[key])
    return [context["entity_id"] for context in contexts], edge_ids


def evaluate_edges(gold: list[dict], edges: list[str], elapsed_ms: float, entities: list[str]) -> dict:
    expected = {edge["id"] for edge in gold}
    hits = expected.intersection(edges)
    top = expected.intersection(edges[:20])
    rank = next((i for i, value in enumerate(edges[:20], 1) if value in expected), None)
    return {"edge_recall": len(hits) / len(expected), "all_edges_found": expected <= set(edges),
            "edge_recall_at_20": len(top) / len(expected), "all_edges_at_20": expected <= set(edges[:20]),
            "first_edge_mrr_at_20": 1/rank if rank else 0.0,
            "gold_source_entity_recall": len({edge["source_id"] for edge in gold}.intersection(entities)) /
                len({edge["source_id"] for edge in gold}),
            "returned_edge_count": len(edges), "latency_ms": round(elapsed_ms, 2),
            "entities": entities, "edge_ids": edges}


def aggregate(rows: list[dict], variant: str) -> dict:
    selected = [row["variants"][variant] for row in rows]
    fields = ("edge_recall", "all_edges_found", "edge_recall_at_20", "all_edges_at_20",
              "first_edge_mrr_at_20", "gold_source_entity_recall", "returned_edge_count", "latency_ms")
    result = {field: round(statistics.mean(item[field] for item in selected), 4) for field in fields}
    result["cases"] = len(selected)
    result["errors"] = sum(bool(item.get("error")) for item in selected)
    paths = [row["variants"][variant] for row in rows if row["type"] == "two_hop"]
    result["two_hop_cases"] = len(paths)
    result["two_hop_all_edges_found"] = round(statistics.mean(item["all_edges_found"] for item in paths), 4) if paths else 0
    result["two_hop_all_edges_at_20"] = round(statistics.mean(item["all_edges_at_20"] for item in paths), 4) if paths else 0
    return result


def main() -> None:
    if torch is None:
        raise SystemExit("Install requirements-graph-benchmark.txt to run this offline relationship experiment.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Isolated evaluation directory, never a serving volume")
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/relationship-questions-v1.json")
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    dataset = json.loads(raw)
    questions = [item for item in dataset["cases"] if item["split"] == args.split]
    if not questions:
        raise SystemExit("The requested split has no cases")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    models = Ollama(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                    os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
                    os.getenv("ROUTER_MODEL", "qwen3:4b"), os.getenv("ANSWER_MODEL", "qwen3:4b"))
    start = time.perf_counter()
    index = SchemaIndex(ROOT / "seed/catalog.json", args.data_dir / "baseline", models)
    baseline_setup = time.perf_counter() - start
    if dataset["source_commit"] != index.catalog["source_commit"]:
        raise SystemExit("The dataset and catalog use different source revisions")
    graph_edges = index.graph.edges
    by_triple = {source_key(edge): edge["id"] for edge in graph_edges.values()}
    if len(by_triple) != len(graph_edges):
        raise SystemExit("Graph triples are not unique")
    for item in questions:
        for gold in item["gold_edges"]:
            if by_triple.get(source_key(gold)) != gold["id"]:
                raise SystemExit(f"Gold edge not in the indexed snapshot: {item['id']}")
    start = time.perf_counter()
    enriched = NeighborhoodIndex(index, models, args.data_dir / "neighborhood-chroma")
    enriched_setup = time.perf_counter() - start
    learned, train_info = train_transe(index.graph)
    explorer = GraphExplorer(index, models, learned)
    # Warm the exact code paths before recording per-query timings.
    warmup_candidates = index.search(questions[0]["question"], limit=20, mode="hybrid")
    enriched.search(questions[0]["question"], limit=20)
    rerank(questions[0]["question"], warmup_candidates, models, 6)
    cases = []
    for case in questions:
        question, gold = case["question"], case["gold_edges"]
        start = time.perf_counter()
        baseline_candidates = index.search(question, limit=20, mode="hybrid")
        search_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        raw_entities, raw_edges = contexts_and_edges(index, question, baseline_candidates, baseline_candidates[:6], by_triple)
        raw_ms = search_ms + (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        error = None
        try:
            baseline_selected = rerank(question, baseline_candidates, models, 6)
        except ModelError as exc:
            baseline_selected, error = [], type(exc).__name__
        baseline_entities, baseline_edges = (contexts_and_edges(index, question, baseline_candidates, baseline_selected, by_triple)
                                             if not error else ([], []))
        baseline_ms = search_ms + (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        enriched_candidates = enriched.search(question, limit=20)
        enriched_error = None
        try:
            enriched_selected = rerank(question, enriched_candidates, models, 6)
        except ModelError as exc:
            enriched_selected, enriched_error = [], type(exc).__name__
        enriched_entities, enriched_edges = (contexts_and_edges(index, question, enriched_candidates, enriched_selected, by_triple)
                                             if not enriched_error else ([], []))
        enriched_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        graph_edges_ranked, transe_edges, graph_info = (explorer.rank(question, baseline_entities) if not error
                                                       else ([], [], {"seed_ids": [], "candidate_edges": 0}))
        graph_extra_ms = (time.perf_counter() - start) * 1000
        graph_entities = list(dict.fromkeys(entity for eid in graph_edges_ranked for entity in
                                            (graph_edges[eid]["source_id"], graph_edges[eid]["target_id"])))
        transe_entities = list(dict.fromkeys(entity for eid in transe_edges for entity in
                                             (graph_edges[eid]["source_id"], graph_edges[eid]["target_id"])))
        measured = {
            "hybrid_raw": evaluate_edges(gold, raw_edges, raw_ms, raw_entities),
            "hybrid_reranked": evaluate_edges(gold, baseline_edges, baseline_ms, baseline_entities),
            "neighborhood_reranked": evaluate_edges(gold, enriched_edges, enriched_ms, enriched_entities),
            "graph_assisted": evaluate_edges(gold, graph_edges_ranked, baseline_ms + graph_extra_ms, graph_entities),
            "graph_transe": evaluate_edges(gold, transe_edges, baseline_ms + graph_extra_ms, transe_entities),
        }
        if error:
            measured["hybrid_reranked"]["error"] = error
            measured["graph_assisted"]["error"] = error
            measured["graph_transe"]["error"] = error
        if enriched_error:
            measured["neighborhood_reranked"]["error"] = enriched_error
        cases.append({**case, "variants": measured, "graph_search": graph_info})
        print(f"{args.split} {len(cases)}/{len(questions)} {case['id']}", flush=True)
    result = {"dataset_id": dataset["dataset_id"], "dataset_status": dataset["status"],
              "dataset_sha256": hashlib.sha256(raw).hexdigest(), "source_commit": index.catalog["source_commit"],
              "split": args.split, "embedding_model": models.embedding_model, "model_digest": index.model_digest,
              "reranker_model": models.answer_model, "baseline_index_revision": index.index_revision,
              "neighborhood_index_revision": enriched.identity,
              "graph": {"nodes": len(index.graph.entities), "edges": len(graph_edges),
                        "relationship_labels": len({edge['attribute'] for edge in graph_edges.values()})},
              "settings": {"candidate_chunks": 20, "reranked_chunks": 6, "expanded_entities": 4,
                           "edge_limit_for_graph_variants": 20, "neighborhood_edges_per_passage": 12,
                           "graph_radius": 2, "rrf_k": 20, "semantic_weight": 1, "lexical_weight": 1,
                           "proximity_weight": 0.5, "transe_weight": 0.5,
                           "query": "raw question; routing, query generation and final answer excluded"},
              "setup_seconds": {"baseline": round(baseline_setup, 2), "neighborhood": round(enriched_setup, 2)},
              "transe": train_info,
              "notes": "Synthetic source-derived labels and questions are not independently reviewed. Unlabeled edges may be relevant. Baseline returns all graph edges in up to four expanded entity contexts; graph variants return a ranked list capped at 20. At-20 metrics compare position, while full-context coverage reflects their different evidence budgets. The TransE model trains on the full observed graph, without query labels; this measures retrieval on known edges, not link prediction or inferred facts. Timings include the same local LLM reranker where applicable but exclude gate, query rewriting, answer generation and network/API overhead.",
              "metrics": {variant: aggregate(cases, variant) for variant in VARIANTS}, "cases": cases,
              "edge_catalog": [{key: edge.get(key) for key in ("id", "source_id", "attribute", "target_id", "description", "source_uri")}
                               for edge in sorted(graph_edges.values(), key=lambda edge: edge["id"])]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": result["metrics"], "transe": train_info}, indent=2), flush=True)


if __name__ == "__main__":
    main()
