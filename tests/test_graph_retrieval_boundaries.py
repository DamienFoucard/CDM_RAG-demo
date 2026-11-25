"""Boundary tests for source-qualified graph evidence and passage retrieval."""

from types import SimpleNamespace

import pytest
from rank_bm25 import BM25Okapi

from app.graph import SchemaGraph, resolve_target
from app.index import SchemaIndex, _split_embedding_text, tokens


def entity(model, name, filename, relationships=()):
    return {
        "id": f"{model}::{name}",
        "model": model,
        "name": name,
        "source_uri": f"https://example.org/schemaDocuments/{model}/{filename}.cdm.json",
        "attributes": [],
        "relationships": list(relationships),
    }


def test_unqualified_relationships_stay_within_the_source_model():
    source = entity("A", "Root", "Root")
    same_model = entity("A", "Shared", "Shared")
    other_model = entity("B", "Shared", "Shared")
    only_elsewhere = entity("B", "External", "External")
    entities = {item["id"]: item for item in (source, same_model, other_model, only_elsewhere)}

    assert resolve_target(entities, source, "Shared") == "A::Shared"
    assert resolve_target(entities, source, "External") is None
    assert resolve_target(entities, source, "B::Shared") == "B::Shared"
    assert resolve_target(entities, source, "../B/Shared.cdm.json/Shared") == "B::Shared"


def test_ambiguous_path_is_reported_as_unresolved_instead_of_inventing_an_edge():
    source = entity("A", "Root", "Root", [
        {"attribute": "ambiguous", "targets": ["../B/Shared.cdm.json"]},
        {"attribute": "explicit", "targets": ["../B/Shared.cdm.json/Second"]},
    ])
    first = entity("B", "First", "Shared")
    second = entity("B", "Second", "Shared")
    graph = SchemaGraph([source, first, second], revision="pinned-revision")

    assert [(edge["attribute"], edge["target_id"]) for edge in graph.edges.values()] == [
        ("explicit", "B::Second")
    ]
    assert [(item["attribute"], item["reference"], item["reason"]) for item in graph.unresolved] == [
        ("ambiguous", "../B/Shared.cdm.json", "unresolved_in_catalog")
    ]


def test_two_hop_evidence_prefers_business_relationships_and_keeps_edge_direction():
    start = entity("M", "Start", "Start", [
        {"attribute": "createdBy", "targets": ["Audit"]},
        {"attribute": "owns", "targets": ["Holding"]},
    ])
    audit = entity("M", "Audit", "Audit", [{"attribute": "modified", "targets": ["End"]}])
    holding = entity("M", "Holding", "Holding", [{"attribute": "details", "targets": ["End"]}])
    end = entity("M", "End", "End")
    index = SchemaIndex.__new__(SchemaIndex)
    index.graph = SchemaGraph([start, audit, holding, end], revision="pinned-revision")

    evidence = index.relationship_evidence("M::End", "M::Start")
    assert evidence["direct"] == []
    assert [(edge["source_id"], edge["attribute"], edge["target_id"]) for edge in evidence["path"]] == [
        ("M::Holding", "details", "M::End"),
        ("M::Start", "owns", "M::Holding"),
    ]
    assert all(edge["source_revision"] == "pinned-revision" for edge in evidence["path"])


@pytest.mark.parametrize("text", [
    "Long sentence. Another sentence; a long attribute name and some final words.",
    "averylongidentifierwithoutanyseparators" * 3,
    "αβγδεζηθικλμνξοπρστυφχψω\n" * 4,
])
def test_embedding_fragments_preserve_all_text_within_the_limit(text):
    fragments = _split_embedding_text(text, max_chars=24)
    assert "".join(fragments) == text
    assert all(0 < len(part) <= 24 for part in fragments)


def test_unknown_vector_fragment_fails_instead_of_returning_unrelated_evidence():
    index = SchemaIndex.__new__(SchemaIndex)
    index.embedding_records = [{"id": "known"}]
    index.vector_to_chunk = {"known": "passage"}
    index.models = SimpleNamespace(embed=lambda _texts, *, query: [[0.1, 0.9]])
    index.collection = SimpleNamespace(query=lambda **_kwargs: {"ids": [["stale-fragment"]]})

    with pytest.raises(RuntimeError, match="unknown CDM passage"):
        index.search("financial holding", mode="semantic")


def test_lexical_search_does_not_call_the_embedding_model():
    index = SchemaIndex.__new__(SchemaIndex)
    index.chunks = [{"id": "account", "text": "account identifier"},
                    {"id": "card", "text": "payment card"}]
    index.by_id = {chunk["id"]: chunk for chunk in index.chunks}
    index.tokenized = [set(tokens(chunk["text"])) for chunk in index.chunks]
    index.bm25 = BM25Okapi([tokens(chunk["text"]) for chunk in index.chunks])
    index.models = SimpleNamespace(embed=lambda *_args, **_kwargs: pytest.fail("lexical search embedded the query"))

    assert [chunk["id"] for chunk in index.search("account", mode="lexical")] == ["account"]
