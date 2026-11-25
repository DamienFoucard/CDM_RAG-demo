from pathlib import Path
from types import SimpleNamespace

import pytest

from app.schema.corpus import load_catalog, render_chunks
from app.schema.graph import SchemaGraph
from app.retrieval.rag import _answer_from_query, answer_question, prepare_passages, selected_entity_definition_response


@pytest.fixture
def catalog_index():
    catalog = load_catalog(Path(__file__).resolve().parents[1] / "seed" / "catalog.json")
    graph = SchemaGraph(catalog["entities"], catalog["source_commit"], semantics=catalog.get("semantics", {}))
    return SimpleNamespace(graph=graph, by_id={c["id"]: c for c in render_chunks(catalog)})


def test_overdraft_definition_uses_exact_selected_source_without_extra_claims(catalog_index, monkeypatch):
    eid = "RetailBankingCoreDataModel::FIOverdraft"
    candidates = [catalog_index.by_id[f"{eid}:entity"],
                  catalog_index.by_id[f"{eid}:attributes:4"],
                  catalog_index.by_id[f"{eid}:attributes:3"]]
    catalog_index.search = lambda *args, **kwargs: candidates
    monkeypatch.setattr("app.retrieval.rag.cited_answer", lambda *args, **kwargs: pytest.fail("unexpected model call"))
    result = _answer_from_query("What is an overdraft in the retail banking model?", "query", catalog_index,
                                SimpleNamespace(), rerank_enabled=False, limit=6, mode="hybrid", generation_limit=4)
    description = catalog_index.graph.entities[eid]["description"]
    assert result["generator"] == "verified-catalog-definition"
    assert result["answer"] == f"{eid}: {description} [R1]"
    assert "overdraft limit" not in result["answer"].lower()
    assert len(result["sources"]) == 1
    assert result["sources"][0]["passage_id"] == f"{eid}:entity"


def test_entity_name_and_model_qualified_name_match_the_catalog(catalog_index):
    eid = "RetailBankingCoreDataModel::FIOverdraft"
    contexts, sources, _ = prepare_passages([catalog_index.by_id[f"{eid}:entity"]], 1)
    assert selected_entity_definition_response("Define FIOverdraft.", catalog_index, contexts, sources)
    assert selected_entity_definition_response(f"What is {eid}?", catalog_index, contexts, sources)


@pytest.mark.parametrize("question", [
    "What is an overdraft and what is its limit?",
    "What is an overdraft, and what is its rate?",
    "What is an overdraft limit?",
    "What are the attributes of an overdraft?",
    "What is an overdraft in the applicationCommon model?",
    "What is an overdraft in the made-up RetailBankingCoreDataModel?",
])
def test_multipart_attribute_and_wrong_model_questions_fall_through(catalog_index, question):
    entity = catalog_index.by_id["RetailBankingCoreDataModel::FIOverdraft:entity"]
    contexts, sources, _ = prepare_passages([entity], 1)
    assert selected_entity_definition_response(question, catalog_index, contexts, sources) is None


def test_ambiguous_account_name_is_not_resolved_by_selected_search_result(catalog_index):
    entity = catalog_index.by_id["applicationCommon::Account:entity"]
    contexts, sources, _ = prepare_passages([entity], 1)
    assert selected_entity_definition_response("What is an Account?", catalog_index, contexts, sources) is None
    answer = selected_entity_definition_response("What is an Account in the applicationCommon model?",
                                                 catalog_index, contexts, sources)
    assert answer is not None and answer[0].startswith("applicationCommon::Account:")


def test_missing_or_altered_entity_passage_never_uses_uncited_catalog_text(catalog_index):
    eid = "RetailBankingCoreDataModel::FIOverdraft"
    attribute = catalog_index.by_id[f"{eid}:attributes:3"]
    contexts, sources, _ = prepare_passages([attribute], 1)
    question = "What is an overdraft in the retail banking model?"
    assert selected_entity_definition_response(question, catalog_index, contexts, sources) is None
    contexts, sources, _ = prepare_passages([catalog_index.by_id[f"{eid}:entity"]], 1)
    contexts[0]["text"] = "Different evidence text"
    assert selected_entity_definition_response(question, catalog_index, contexts, sources) is None


def test_verified_definition_skips_optional_model_judge(catalog_index, monkeypatch):
    eid = "RetailBankingCoreDataModel::FIOverdraft"
    catalog_index.search = lambda *args, **kwargs: [catalog_index.by_id[f"{eid}:entity"]]
    catalog_index.semantic_probe = lambda question: 1.0
    catalog_index.index_revision = "fixture"
    catalog_index.catalog = {"source_commit": "fixture"}
    monkeypatch.setattr("app.retrieval.rag.assess_scope", lambda *args: "schema")
    monkeypatch.setattr("app.retrieval.rag.plan_routed_search", lambda *args: (True, "overdraft", {}))
    monkeypatch.setattr("app.retrieval.rag.review_answer", lambda *args: pytest.fail("unexpected model judge call"))
    result = answer_question("What is an overdraft in the retail banking model?", catalog_index,
                             SimpleNamespace(), judge_enabled=True)
    assert result["answer_quality"]["status"] == "verified_catalog_definition"
    assert result["answer_quality"]["judge_calls"] == 0
    assert result["retrieval"]["generator"] == "verified-catalog-definition"
