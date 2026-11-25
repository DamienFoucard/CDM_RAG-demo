from types import SimpleNamespace

from app.retrieval.rag import format_entity_citations, named_relationship_answer, no_direct_relationship_answer


def edge(edge_id, source, attribute, target):
    return {"id": edge_id, "source_id": source, "attribute": attribute,
            "target_id": target, "source_uri": "https://example.org/shared.cdm.json"}


def test_parallel_edges_from_one_definition_have_individual_supporting_passages():
    first = edge("1", "M::Account", "primaryContact", "M::Contact")
    second = edge("2", "M::Account", "billingContact", "M::Contact")
    index = SimpleNamespace(relationship_evidence=lambda *_: {"direct": [first, second], "path": []})
    answer, sources = no_direct_relationship_answer(index, "M::Account", "M::Contact", [])

    assert "M::Account --primaryContact--> M::Contact [R1]" in answer
    assert "M::Account --billingContact--> M::Contact [R2]" in answer
    assert [source["text"] for source in sources] == [
        "M::Account --primaryContact--> M::Contact.",
        "M::Account --billingContact--> M::Contact.",
    ]
    formatted, displayed = format_entity_citations(answer, sources)
    assert "[M::Account · R1]" in formatted and "[M::Account · R2]" in formatted
    assert len(displayed) == 2


def test_two_hop_path_with_shared_source_retains_both_edge_descriptions():
    first = edge("1", "M::Bridge", "first", "M::A")
    second = edge("2", "M::Bridge", "second", "M::B")
    index = SimpleNamespace(relationship_evidence=lambda *_: {"direct": [], "path": [first, second]})
    answer, sources = no_direct_relationship_answer(index, "M::A", "M::B", [])

    assert "M::A <--first-- M::Bridge --second--> M::B" in answer
    assert "[R1]" in answer and "[R2]" in answer
    assert [source["text"] for source in sources] == [
        "M::Bridge --first--> M::A.", "M::Bridge --second--> M::B.",
    ]


def test_named_edges_never_reuse_an_unrelated_existing_passage_or_label():
    first = edge("1", "M::Account", "primaryContact", "M::Contact")
    second = edge("2", "M::Account", "billingContact", "M::Contact")
    index = SimpleNamespace(graph=SimpleNamespace(adjacency={"M::Account": [first, second]}))
    existing = [{"citation": "R2", "entity_id": "M::Account", "uri": first["source_uri"],
                 "kind": "entity", "text": "An account definition, without these relationships."}]
    answer, sources = named_relationship_answer(index, "M::Account", ["M::Contact"], existing)

    assert "--billingContact--> M::Contact [R1]" in answer
    assert "--primaryContact--> M::Contact [R3]" in answer
    assert {source["citation"] for source in sources} == {"R1", "R3"}
    assert all(source["kind"] == "relationship" for source in sources)
    assert {source["text"] for source in sources} == {
        "M::Account --billingContact--> M::Contact.",
        "M::Account --primaryContact--> M::Contact.",
    }
