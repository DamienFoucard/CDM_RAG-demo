from types import SimpleNamespace
from pathlib import Path

from app.schema.corpus import load_catalog, render_chunks
from app.schema.graph import SchemaGraph
from app.retrieval.rag import (_answer_from_query, prepare_passages, select_relationship_attribute_evidence,
                     shared_target_relationship_response)


def make_index(*, duplicate=False, unresolved=False):
    bank_id = "RetailBankingCoreDataModel::FICard"
    common_id = "applicationCommon::Card"
    entities = {
        bank_id: {"name": "FICard", "model": "RetailBankingCoreDataModel",
                  "relationships": [{"attribute": "Cardholder"}]},
        "RetailBankingCoreDataModel::CustomerFinancialHolding": {
            "name": "CustomerFinancialHolding", "model": "RetailBankingCoreDataModel",
            "relationships": [{"attribute": "Customer"}]},
    }
    if duplicate:
        entities[common_id] = {"name": "Card", "model": "applicationCommon",
                               "relationships": [{"attribute": "Cardholder"}]}
    edges = {} if unresolved else {"card": {"source_id": bank_id, "attribute": "Cardholder"}}
    edges["customer"] = {"source_id": "RetailBankingCoreDataModel::CustomerFinancialHolding",
                         "attribute": "Customer"}
    if duplicate:
        edges["common"] = {"source_id": common_id, "attribute": "Cardholder"}
    by_id = {f"{entity_id}:relationship:1": {"id": f"{entity_id}:relationship:1",
             "entity_id": entity_id, "kind": "relationship", "text": "explicit source relationship",
             "uri": "https://example.com/source"} for entity_id in entities}
    return SimpleNamespace(graph=SimpleNamespace(entities=entities, edges=edges), by_id=by_id)


def candidate(entity_id, number):
    return {"id": f"p{number}", "entity_id": entity_id, "kind": "attributes",
            "text": "source attributes", "uri": "https://example.com/source"}


def test_cardholder_reference_reserves_one_slot_without_reordering_ranked_evidence():
    index = make_index()
    financial = "RetailBankingCoreDataModel::CustomerFinancialHolding"
    card = "RetailBankingCoreDataModel::FICard"
    candidates = [candidate(financial, i) for i in range(4)] + [candidate(card, 4)]
    question = "In the banking model, what can a cardholder refer to, and how does that relate to the customer linked to a financial holding?"
    selected = select_relationship_attribute_evidence(question, index, candidates, candidates, 4)
    assert [x["id"] for x in selected] == ["p0", "p1", "p2", f"{card}:relationship:1"]


def test_ambiguous_name_is_not_resolved_by_candidate_rank():
    index = make_index(duplicate=True)
    card = "RetailBankingCoreDataModel::FICard"
    candidates = [candidate(card, i) for i in range(4)]
    question = "What can the cardholder reference?"
    assert select_relationship_attribute_evidence(question, index, candidates, candidates, 4) == candidates
    constrained = "In the RetailBankingCoreDataModel, what can the cardholder reference?"
    selected = select_relationship_attribute_evidence(constrained, index, candidates, candidates, 4)
    assert selected[-1]["id"] == f"{card}:relationship:1"


def test_model_constraint_does_not_select_another_models_field():
    index = make_index()
    card = "RetailBankingCoreDataModel::FICard"
    candidates = [candidate(card, i) for i in range(4)]
    question = "In the applicationCommon model, what can the cardholder refer to?"
    # The explicitly named model must be recognized even if it has no loaded entity.
    index.graph.entities["applicationCommon::Contact"] = {
        "name": "Contact", "model": "applicationCommon", "relationships": []}
    assert select_relationship_attribute_evidence(question, index, candidates, candidates, 4) == candidates


def test_generic_customer_word_needs_explicit_field_syntax():
    index = make_index()
    financial = "RetailBankingCoreDataModel::CustomerFinancialHolding"
    candidates = [candidate(financial, i) for i in range(4)]
    assert select_relationship_attribute_evidence(
        "How is a customer linked to a holding?", index, candidates, candidates, 4) == candidates
    selected = select_relationship_attribute_evidence(
        "What can the field Customer reference?", index, candidates, candidates, 4)
    assert selected[-1]["id"] == f"{financial}:relationship:1"


def test_does_not_inject_without_candidate_source_or_resolved_edge():
    card = "RetailBankingCoreDataModel::FICard"
    other = "RetailBankingCoreDataModel::CustomerFinancialHolding"
    candidates = [candidate(other, i) for i in range(4)]
    assert select_relationship_attribute_evidence(
        "What can the cardholder reference?", make_index(), candidates, candidates, 4) == candidates
    candidates = [candidate(card, i) for i in range(4)]
    assert select_relationship_attribute_evidence(
        "What can the cardholder reference?", make_index(unresolved=True), candidates, candidates, 4) == candidates
    assert select_relationship_attribute_evidence(
        "What does cardholder mean?", make_index(), candidates, candidates, 4) == candidates


def test_real_catalog_cardholder_selection_keeps_specific_source_and_targets():
    catalog = load_catalog(Path(__file__).resolve().parents[1] / "seed" / "catalog.json")
    graph = SchemaGraph(catalog["entities"], catalog["source_commit"], semantics=catalog.get("semantics", {}))
    index = SimpleNamespace(graph=graph, by_id={item["id"]: item for item in render_chunks(catalog)})
    sources = {edge["source_id"] for edge in graph.edges.values() if edge["attribute"].casefold() == "cardholder"}
    assert sources == {"RetailBankingCoreDataModel::FICard"}
    ids = ["RetailBankingCoreDataModel::CustomerFinancialHolding:attributes:3",
           "RetailBankingCoreDataModel::CustomerFinancialHolding:relationship:2",
           "RetailBankingCoreDataModel::CustomerFinancialHolding:relationship:1",
           "RetailBankingCoreDataModel::CustomerFinancialHolding:attributes:2",
           "RetailBankingCoreDataModel::FICard:attributes:5",
           "RetailBankingCoreDataModel::FICard:relationship:1"]
    candidates = [index.by_id[item] for item in ids]
    question = ("In the banking model, what can a cardholder refer to, and how does that relate "
                "to the customer linked to a financial holding?")
    selected = select_relationship_attribute_evidence(question, index, candidates, candidates, 4)
    assert [item["id"] for item in selected] == [*ids[:3], ids[-1]]
    assert "RetailBankingCoreDataModel::Account" in selected[-1]["text"]
    assert "RetailBankingCoreDataModel::Contact" in selected[-1]["text"]


def test_serving_path_passes_reserved_evidence_to_answer_model(monkeypatch):
    index = make_index()
    financial = "RetailBankingCoreDataModel::CustomerFinancialHolding"
    card = "RetailBankingCoreDataModel::FICard"
    candidates = [candidate(financial, i) for i in range(4)] + [candidate(card, 4)]
    index.search = lambda *args, **kwargs: candidates
    seen = []

    def fake_answer(question, sources, contexts, models, feedback, **kwargs):
        seen.extend(contexts)
        return "answer", "fake"

    monkeypatch.setattr("app.retrieval.rag.cited_answer", fake_answer)
    result = _answer_from_query(
        "What can the cardholder refer to?", "search", index, SimpleNamespace(),
        rerank_enabled=False, limit=6, mode="hybrid", generation_limit=4,
    )
    assert len(result["contexts"]) == 4
    assert seen[-1]["passage_id"] == f"{card}:relationship:1"


def real_relationship_fixture():
    catalog = load_catalog(Path(__file__).resolve().parents[1] / "seed" / "catalog.json")
    graph = SchemaGraph(catalog["entities"], catalog["source_commit"], semantics=catalog.get("semantics", {}))
    index = SimpleNamespace(graph=graph, by_id={item["id"]: item for item in render_chunks(catalog)})
    ids = ["RetailBankingCoreDataModel::CustomerFinancialHolding:attributes:3",
           "RetailBankingCoreDataModel::CustomerFinancialHolding:relationship:2",
           "RetailBankingCoreDataModel::CustomerFinancialHolding:relationship:1",
           "RetailBankingCoreDataModel::CustomerFinancialHolding:attributes:2",
           "RetailBankingCoreDataModel::FICard:attributes:5",
           "RetailBankingCoreDataModel::FICard:relationship:1"]
    candidates = [index.by_id[item] for item in ids]
    index.search = lambda *args, **kwargs: candidates
    return index, candidates


def test_serving_shared_target_response_cites_all_three_edges_without_equating_records(monkeypatch):
    index, _ = real_relationship_fixture()
    question = ("In the banking model, what can a cardholder refer to, and how does that relate "
                "to the customer linked to a financial holding?")
    def must_not_generate(*args, **kwargs):
        raise AssertionError("The shared-target answer must not ask the model to infer an identity")
    monkeypatch.setattr("app.retrieval.rag.cited_answer", must_not_generate)
    result = _answer_from_query(question, "search", index, SimpleNamespace(),
                                rerank_enabled=False, limit=6, mode="hybrid", generation_limit=4)
    assert result["generator"] == "verified-graph-response"
    assert "FICard.Cardholder references" in result["answer"]
    assert "CustomerFinancialHolding.Customer references" in result["answer"]
    assert "CustomerFinancialHolding.FinancialHolding references" in result["answer"]
    assert "do not establish that the referenced records are the same" in result["answer"]
    assert len(result["sources"]) == 3
    assert all(f"[{source['citation']}]" in result["answer"] for source in result["sources"])


def test_shared_target_response_requires_relevant_source_and_does_not_overblock_definitions():
    index, candidates = real_relationship_fixture()
    contexts, sources, _ = prepare_passages([candidates[1], candidates[2], candidates[5]], 3)
    assert shared_target_relationship_response(
        "How does a cardholder relate to a customer?", index, contexts, sources) is None
    assert shared_target_relationship_response(
        "What is a cardholder in a financial holding?", index, contexts, sources) is None


def test_distinct_target_types_do_not_trigger_shared_target_response():
    index, candidates = real_relationship_fixture()
    contexts, sources, _ = prepare_passages([candidates[1], candidates[2], candidates[5]], 3)
    for edge in index.graph.edges.values():
        if edge.get("source_id") == "RetailBankingCoreDataModel::FICard" and edge.get("attribute") == "Cardholder":
            edge["target_id"] = "applicationCommon::Contact"
    question = ("In the banking model, what can a cardholder refer to, and how does that relate "
                "to the customer linked to a financial holding?")
    assert shared_target_relationship_response(question, index, contexts, sources) is None
