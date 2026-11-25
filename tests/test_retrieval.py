from __future__ import annotations

import hashlib
import json
from pathlib import Path
from rank_bm25 import BM25Okapi

from app.retrieval.index import SchemaIndex, tokens
from app.runtime.ollama import ModelError, Ollama
import pytest


class FakeCollection:
    def query(self, **_kwargs):
        return {"ids": [["semantic::fragment:1", "semantic::fragment:2", "lexical"]],
                "distances": [[0.2, 0.3, 0.6]]}


class FakeModels:
    def embed(self, _texts, *, query):
        assert query
        return [[0.1, 0.9]]


def test_hybrid_search_combines_semantic_and_lexical_rankings():
    index = SchemaIndex.__new__(SchemaIndex)
    index.chunks = [
        {"id": "lexical", "entity_id": "M::A", "text": "account number identifier"},
        {"id": "semantic", "entity_id": "M::B", "text": "customer banking record"},
        {"id": "other", "entity_id": "M::C", "text": "unrelated information"},
    ]
    index.by_id = {item["id"]: item for item in index.chunks}
    index.embedding_records = [{"id": item} for item in
                               ("semantic::fragment:1", "semantic::fragment:2", "lexical")]
    index.vector_to_chunk = {"semantic::fragment:1": "semantic", "semantic::fragment:2": "semantic",
                             "lexical": "lexical"}
    index.tokenized = [set(tokens(item["text"])) for item in index.chunks]
    index.bm25 = BM25Okapi([tokens(item["text"]) for item in index.chunks])
    index.collection = FakeCollection()
    index.models = FakeModels()

    assert index.semantic_probe("account") == 0.8
    result = index.search("account identifier", limit=3, mode="hybrid")
    assert result[0]["id"] == "lexical"
    assert {item["id"] for item in result} == {"lexical", "semantic"}
    assert [item["id"] for item in index.search("account identifier", mode="lexical")] == ["lexical"]
    assert index.search("", mode="hybrid") == []


def test_bundled_source_inputs_match_their_recorded_hashes():
    root = Path(__file__).parents[1] / "seed"
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    files = catalog["provenance"]["files"]
    assert len(catalog["entities"]) == 64
    assert files
    for item in files:
        path = root / "source" / item["path"]
        assert path.is_file(), item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_embedding_models_use_their_own_query_and_document_formats():
    received = []
    for name in ("nomic-embed-text", "qwen3-embedding:0.6b"):
        model = Ollama("http://localhost", name, "router", "answer")
        model._post = lambda _path, payload: received.append(payload["input"]) or {"embeddings": [[0.1]]}
        model.embed(["query"], query=True)
        model.embed(["document"], query=False)
    assert received == [["search_query: query"], ["search_document: document"],
                        ["Instruct: Retrieve passages from the extracted Microsoft CDM catalog that answer the user's question.\nQuery: query"],
                        ["document"]]
    with pytest.raises(ModelError, match="no configured"):
        Ollama("http://localhost", "unknown", "router", "answer").embedding_format()
