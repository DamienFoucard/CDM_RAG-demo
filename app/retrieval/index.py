"""A local, persistent Chroma index and matching BM25 index for the CDM snapshot."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.schema.corpus import load_catalog, render_chunks
from app.schema.graph import SchemaGraph
from app.runtime.ollama import Ollama


EMBEDDING_FRAGMENT_CHARS = 3000


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())


def _split_embedding_text(text: str, max_chars: int) -> list[str]:
    """Split long passages without dropping text or exceeding the embedder limit."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        lower_bound = max_chars // 2
        boundaries = [remaining.rfind(marker, lower_bound, max_chars)
                      for marker in ("\n", ". ", "; ", " ")]
        cut = max(boundaries)
        if cut < lower_bound:
            cut = max_chars
        else:
            marker = next((marker for marker in ("\n", ". ", "; ", " ")
                           if remaining.rfind(marker, lower_bound, max_chars) == cut), "")
            cut += len(marker)
        pieces.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        pieces.append(remaining)
    return pieces


class SchemaIndex:
    def __init__(self, catalog_path: Path, data_dir: Path, models: Ollama):
        self.models = models
        self.catalog = load_catalog(catalog_path)
        self.graph = SchemaGraph(self.catalog["entities"], self.catalog["source_commit"],
                                 semantics=self.catalog.get("semantics", {}))
        self.chunks = render_chunks(self.catalog)
        self.by_id = {chunk["id"]: chunk for chunk in self.chunks}
        self.embedding_records: list[dict[str, str]] = []
        self.vector_to_chunk: dict[str, str] = {}
        for chunk in self.chunks:
            text = chunk["text"]
            if len(text) <= EMBEDDING_FRAGMENT_CHARS:
                parts = [text]
            else:
                prefix = f"{chunk.get('title') or chunk['id']}: "
                prefix = prefix[:EMBEDDING_FRAGMENT_CHARS // 4]
                parts = [prefix + part for part in _split_embedding_text(
                    text, EMBEDDING_FRAGMENT_CHARS - len(prefix))]
            for number, part in enumerate(parts, 1):
                vector_id = chunk["id"] if len(parts) == 1 else f"{chunk['id']}::fragment:{number}"
                self.embedding_records.append({"id": vector_id, "chunk_id": chunk["id"], "text": part})
                self.vector_to_chunk[vector_id] = chunk["id"]
        self.bm25 = BM25Okapi([tokens(chunk["text"]) for chunk in self.chunks])
        self.tokenized = [set(tokens(chunk["text"])) for chunk in self.chunks]
        model_digest = models.embedding_digest()
        identity = hashlib.sha256(json.dumps({"chunks": self.chunks, "model": models.embedding_model,
                                              "model_digest": model_digest, "format": "cdm-passages-v3",
                                              "embedding_fragment_chars": EMBEDDING_FRAGMENT_CHARS,
                                              "embedding_format": models.embedding_format()},
                                             sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.index_revision = identity
        self.model_digest = model_digest

        import chromadb

        data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(data_dir / "chroma"))
        name = "cdm_" + identity[:24]
        collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine", "model_digest": model_digest}, embedding_function=None,
        )
        actual = set(collection.get(include=[]).get("ids", []))
        expected = set(self.vector_to_chunk)
        self.index_reused = actual == expected
        if actual != expected:
            # A crashed first build is never considered complete.
            self.client.delete_collection(name)
            collection = self.client.create_collection(
                name=name, metadata={"hnsw:space": "cosine", "model_digest": model_digest},
                embedding_function=None,
            )
            for start in range(0, len(self.embedding_records), 24):
                batch = self.embedding_records[start:start + 24]
                vectors = models.embed([record["text"] for record in batch], query=False)
                source_chunks = [self.by_id[record["chunk_id"]] for record in batch]
                collection.add(ids=[record["id"] for record in batch], embeddings=vectors,
                               documents=[record["text"] for record in batch],
                               metadatas=[{**{key: chunk[key] for key in (
                                   "entity_id", "source_id", "source_type", "source_label", "kind",
                                   "title", "uri", "path", "source_path")
                                   if isinstance(chunk.get(key), str) and chunk[key]},
                                   "source_chunk_id": chunk["id"]}
                                   for chunk in source_chunks])
            if collection.count() != len(self.embedding_records):
                raise RuntimeError("The vector index build did not cover all CDM passages.")
        self.collection = collection

    def semantic_probe(self, question: str) -> float:
        """Return the closest raw cosine similarity without exposing passages."""
        vector = self.models.embed([question], query=True)[0]
        result = self.collection.query(query_embeddings=[vector], n_results=1, include=["distances"])
        distances = result.get("distances") or []
        if not distances or not distances[0]:
            return 0.0
        return max(-1.0, min(1.0, 1.0 - float(distances[0][0])))

    def entity_context(self, entity_id: str) -> dict[str, Any]:
        """Return the complete extracted definition and its direct loaded-catalog neighbors."""
        entity = self.graph.entities[entity_id]
        neighbors = []
        for edge in self.graph.adjacency[entity_id]:
            outgoing = edge["source_id"] == entity_id
            other_id = edge["target_id"] if outgoing else edge["source_id"]
            other = self.graph.entities[other_id]
            neighbors.append({"direction": "outgoing" if outgoing else "incoming",
                              "attribute": edge["attribute"], "entity_id": other_id,
                              "source_entity_id": edge["source_id"], "target_entity_id": edge["target_id"],
                              "entity_name": other["name"], "description": edge.get("description", ""),
                              "source_uri": edge["source_uri"]})
        return {"entity_id": entity_id, "name": entity["name"], "model": entity["model"],
                "description": entity.get("description", ""), "source_uri": entity["source_uri"],
                "attributes": [{"name": attribute["name"], "data_type": attribute.get("data_type"),
                                "description": attribute.get("description", "")}
                               for attribute in entity.get("attributes", [])],
                "direct_neighbors": neighbors,
                "unresolved_references": [item for item in self.graph.unresolved if item["source_id"] == entity_id],
                "coverage": "extracted references within the loaded CDM subset"}

    def relationship_evidence(self, first: str, second: str) -> dict[str, Any]:
        """Find direct edges and a bounded, explicitly labeled two-hop path."""
        direct = [edge for edge in self.graph.adjacency[first]
                  if {edge["source_id"], edge["target_id"]} == {first, second}]
        paths = []
        for edge in self.graph.adjacency[first]:
            middle = edge["target_id"] if edge["source_id"] == first else edge["source_id"]
            if middle in {first, second}:
                continue
            for next_edge in self.graph.adjacency[middle]:
                target = next_edge["target_id"] if next_edge["source_id"] == middle else next_edge["source_id"]
                if target == second:
                    paths.append([edge, next_edge])

        def path_priority(path: list[dict[str, Any]]) -> tuple:
            # Administrative audit links are real but rarely explain the
            # business relationship the question is asking about.
            names = " ".join(edge["attribute"].lower() for edge in path)
            administrative = sum(term in names for term in ("created", "modified", "onbehalf"))
            return administrative, tuple(edge["id"] for edge in path)

        paths.sort(key=path_priority)
        return {"direct": direct, "path": paths[0] if paths else []}

    def search(self, question: str, *, limit: int = 24, mode: str = "hybrid") -> list[dict[str, Any]]:
        if mode not in {"hybrid", "semantic", "lexical"} or not 1 <= limit <= 100:
            raise ValueError("unsupported retrieval mode or limit")
        query_tokens = tokens(question)
        if not query_tokens:
            return []
        lexical_ids: list[str] = []
        vector_ids: list[str] = []
        if mode in {"hybrid", "lexical"}:
            scores = self.bm25.get_scores(query_tokens)
            indices = [index for index, words in enumerate(self.tokenized) if words.intersection(query_tokens)]
            indices.sort(key=lambda index: (-float(scores[index]), self.chunks[index]["id"]))
            lexical_ids = [self.chunks[index]["id"] for index in indices[:limit]]
        if mode in {"hybrid", "semantic"}:
            vector = self.models.embed([question], query=True)[0]
            candidate_count = min(max(limit * 4, 100), len(self.embedding_records))
            result = self.collection.query(query_embeddings=[vector], n_results=candidate_count)
            fragment_ids = result["ids"][0]
            if any(item not in self.vector_to_chunk for item in fragment_ids):
                raise RuntimeError("The vector index returned an unknown CDM passage.")
            vector_ids = list(dict.fromkeys(self.vector_to_chunk[item] for item in fragment_ids))
        lexical_ranks = {item: rank for rank, item in enumerate(lexical_ids, 1)}
        vector_ranks = {item: rank for rank, item in enumerate(vector_ids, 1)}
        ids = list(dict.fromkeys(lexical_ids + vector_ids))
        score = lambda item: ((1 / (60 + lexical_ranks[item])) if item in lexical_ranks else 0) + (
            (1 / (60 + vector_ranks[item])) if item in vector_ranks else 0)
        ids.sort(key=lambda item: (-score(item), item))
        return [{**self.by_id[item], "score": round(score(item), 6)} for item in ids[:limit]]
