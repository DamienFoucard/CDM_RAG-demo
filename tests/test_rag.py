from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.corpus import render_chunks
from app.ollama import ModelError
from app.rag import (answer_question, cited_answer, prepare_passages,
                     format_entity_citations, no_direct_relationship_answer, plan_search, rerank,
                     review_answer, qualified_relationship_entities,
                     named_relationship_answer, _answer_from_query)


class FakeModels:
    router_model = "router"
    answer_model = "answer"
    scope_model = "scope"
    judge_model = "judge"

    def __init__(self, outputs=None, answer="A supported answer.", claims=None, scope="schema",
                 judge_decisions=None, answer_claims=None):
        self.outputs = iter(outputs or [])
        self.answer = answer
        self.claims = claims
        self.answer_claim_sets = iter(answer_claims or [])
        self.judge_decisions = iter(judge_decisions or [])
        self.scope = scope
        self.calls = []
        self.messages = []

    def chat_json(self, model, messages, **kwargs):
        self.calls.append(model)
        self.messages.append((model, messages))
        if model == self.scope_model:
            if isinstance(self.scope, Exception):
                raise self.scope
            return {"requested_information": "question intent", "category": self.scope}
        if model == self.judge_model:
            result = next(self.judge_decisions, accepted_judge())
            if isinstance(result, Exception):
                raise result
            return result
        if model == self.answer_model and kwargs.get("schema", {}).get("required") == ["claims"]:
            default = self.claims if self.claims is not None else [
                {"text": self.answer, "citations": ["R1"]}]
            return {"claims": next(self.answer_claim_sets, default)}
        if model == self.answer_model and kwargs.get("schema"):
            return {}
        return next(self.outputs)

    def chat_json_timed(self, model, messages, **kwargs):
        return self.chat_json(model, messages, **kwargs), None

    def chat(self, model, messages):
        self.calls.append(model)
        return self.answer

    def chat_stream(self, model, messages, *, on_token, **_kwargs):
        self.calls.append(model)
        self.messages.append((model, messages))
        default = self.claims if self.claims is not None else [
            {"text": self.answer, "citations": ["R1"]}]
        claims = next(self.answer_claim_sets, default)
        answer = " ".join(
            claim["text"] + ((" [" + ", ".join(claim.get("citations", [])) + "]")
                              if claim.get("citations") else "")
            for claim in claims
        )
        on_token(answer)
        return answer


class FakeIndex:
    index_revision = "test-index"
    catalog = {"source_commit": "test-source"}

    def __init__(self, similarity=0.7):
        self.similarity = similarity
        self.search_calls = []
        self.graph = SimpleNamespace(entities={"M::Account": {"name": "Account"}})

    def semantic_probe(self, _question):
        return self.similarity

    def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return [{"id": "passage-1", "entity_id": "M::Account", "kind": "attributes",
                 "text": "Account attributes", "uri": "https://example.org/a"}]

    def entity_context(self, entity_id):
        return {"entity_id": entity_id, "name": "Account", "model": "M", "description": "A definition",
                "source_uri": "https://example.org/a", "attributes": [{"name": "id"}],
                "direct_neighbors": [{"source_entity_id": "M::Other", "target_entity_id": entity_id,
                                      "entity_id": "M::Other", "attribute": "account",
                                      "source_uri": "https://example.org/other"}]}


def accepted_judge():
    return {"decision": "accept", "issue": "none", "feedback": "", "search_query": "",
            "claim_checks": [{"claim": "The answer is supported.", "assessment": "supported"}],
            "coverage": "answered", "verdict": "acceptable"}


def answer_retry(feedback="Name the identifier attribute."):
    return {"decision": "retry_answer", "issue": "missing_detail", "feedback": feedback, "search_query": "",
            "claim_checks": [{"claim": "The answer omits a requested detail.", "assessment": "unsupported"}],
            "coverage": "answered", "verdict": "unacceptable"}


def retrieval_retry(query="Account identifier attribute"):
    return {"decision": "retry_retrieval", "issue": "missing_evidence",
            "feedback": "Find the Account identifier definition.", "search_query": query,
            "claim_checks": [{"claim": "The requested detail is not present.", "assessment": "uncertain"}],
            "coverage": "missing", "verdict": "unacceptable"}


def test_low_similarity_skips_planning_and_never_exposes_sources(monkeypatch):
    monkeypatch.setenv("CDM_RAG_MIN_SIMILARITY", "0.6")
    models, index = FakeModels(), FakeIndex(similarity=0.4)
    result = answer_question("My dog Luna likes walks", index, models)
    assert result["route"] == "no_search"
    assert result["sources"] == [] and result["context"] == []
    assert result["retrieval"]["query_generated"] is False
    assert models.calls == [] and index.search_calls == []


def test_default_probe_threshold_depends_on_embedding_model(monkeypatch):
    monkeypatch.delenv("CDM_RAG_MIN_SIMILARITY", raising=False)
    model = FakeModels(outputs=[{"search": False, "query": ""}])
    model.embedding_model = "qwen3-embedding:0.6b"
    accepted = answer_question("Find accounts", FakeIndex(similarity=0.59), model)
    assert accepted["retrieval"]["threshold"] == 0.58
    assert model.calls == ["scope", "router"]
    model = FakeModels()
    model.embedding_model = "nomic-embed-text"
    rejected = answer_question("Find accounts", FakeIndex(similarity=0.59), model)
    assert rejected["retrieval"]["threshold"] == 0.60
    assert model.calls == []


def test_router_can_decline_even_when_similarity_is_high(monkeypatch):
    monkeypatch.setenv("CDM_RAG_MIN_SIMILARITY", "0.6")
    models = FakeModels(outputs=[{"search": False, "query": ""}])
    index = FakeIndex(similarity=0.8)
    result = answer_question("My dog likes the bank", index, models)
    assert result["route"] == "no_search" and result["sources"] == []
    assert result["retrieval"]["query_generated"] is False
    assert index.search_calls == []


def test_attribute_question_uses_the_same_retrieved_passage_answer_path(monkeypatch):
    monkeypatch.setenv("CDM_RAG_MIN_SIMILARITY", "0.6")
    models = FakeModels(outputs=[{"search": True, "query": "Account attributes"}],
                        answer="Account attributes are defined in the retrieved passage.")
    result = answer_question("What are the core attributes of Account?", FakeIndex(), models,
                             judge_enabled=False)
    assert "retrieved passage" in result["answer"]
    assert len(result["context"]) == 1
    context = result["context"][0]
    assert {key: context[key] for key in ("passage_id", "entity_id", "kind", "text", "citation")} == {
        "passage_id": "passage-1", "entity_id": "M::Account", "kind": "attributes",
        "text": "Account attributes", "citation": "R1"}
    assert models.calls[-1] == models.answer_model


def test_generation_preparation_runs_after_retrieval_before_answer():
    models = FakeModels(outputs=[{"search": True, "query": "Account attributes"}])
    index = FakeIndex()

    def prepare():
        assert index.search_calls
        assert models.answer_model not in models.calls
        models.calls.append("unloaded")

    answer_question("What are the core attributes of Account?", index, models,
                    judge_enabled=False, before_generation=prepare)
    assert models.calls[-2:] == ["unloaded", models.answer_model]


def test_generation_preparation_failure_does_not_call_answer_model():
    models = FakeModels(outputs=[{"search": True, "query": "Account attributes"}])

    def prepare():
        raise ModelError("Could not unload model")

    with pytest.raises(ModelError, match="Could not unload"):
        answer_question("What are the core attributes of Account?", FakeIndex(), models,
                        judge_enabled=False, before_generation=prepare)
    assert models.answer_model not in models.calls


def test_qualified_relationship_names_resolve_the_requested_model_and_neighbor_names():
    graph = SimpleNamespace(entities={
        "applicationCommon::Account": {"name": "Account"},
        "RetailBankingCoreDataModel::Account": {"name": "Account"},
        "applicationCommon::Contact": {"name": "Contact"},
        "RetailBankingCoreDataModel::Contact": {"name": "Contact"},
        "applicationCommon::Email": {"name": "Email"},
        "applicationCommon::ActivityParty": {"name": "ActivityParty"},
    })
    index = SimpleNamespace(graph=graph)
    ids = qualified_relationship_entities(
        "List relationships declared by applicationCommon::Account to Contact, Email, ActivityParty.", index)
    assert ids[0] == "applicationCommon::Account"
    assert "RetailBankingCoreDataModel::Account" not in ids
    assert {"applicationCommon::Contact", "applicationCommon::Email",
            "applicationCommon::ActivityParty"}.issubset(ids)
    assert "RetailBankingCoreDataModel::Contact" not in ids


def test_relationship_word_does_not_shadow_entity_names_and_candidates_resolve_variants():
    entities = {
        "RetailBankingCoreDataModel::Relationship": {"name": "Relationship"},
        "applicationCommon::Account": {"name": "Account"},
        "RetailBankingCoreDataModel::Account": {"name": "Account"},
        "applicationCommon::Contact": {"name": "Contact"},
        "RetailBankingCoreDataModel::Contact": {"name": "Contact"},
    }
    index = SimpleNamespace(graph=SimpleNamespace(entities=entities))
    candidates = [
        {"entity_id": "applicationCommon::Contact"},
        {"entity_id": "applicationCommon::Account"},
        {"entity_id": "applicationCommon::Contact"},
        {"entity_id": "applicationCommon::Contact"},
        {"entity_id": "applicationCommon::Contact"},
        {"entity_id": "applicationCommon::Contact"},
        {"entity_id": "RetailBankingCoreDataModel::Relationship"},
    ]
    ids = qualified_relationship_entities(
        "What is the relationship between Account and Contact?", index, candidates)
    assert ids == ["applicationCommon::Account", "applicationCommon::Contact"]


def test_named_relationship_answer_keeps_incoming_and_outgoing_edge_directions():
    account = "applicationCommon::Account"
    contact = "applicationCommon::Contact"
    email = "applicationCommon::Email"
    edges = [
        {"id": "edge-1", "source_id": account, "target_id": contact, "attribute": "primaryContact",
         "source_uri": "https://example.org/account"},
        {"id": "edge-2", "source_id": email, "target_id": account, "attribute": "Sender_emailSender",
         "source_uri": "https://example.org/email"},
    ]
    index = SimpleNamespace(graph=SimpleNamespace(adjacency={account: edges}))
    answer, sources = named_relationship_answer(index, account, [contact, email], [])
    assert f"Outgoing: {account} --primaryContact--> {contact}" in answer
    assert f"Incoming: {email} --Sender_emailSender--> {account}" in answer
    assert f"{account} --Sender_emailSender--> {email}" not in answer
    assert {source["entity_id"] for source in sources} == {account, email}


def test_multi_entity_relationship_request_uses_graph_edges_without_reranking_or_answer_generation():
    account = "applicationCommon::Account"
    contact = "applicationCommon::Contact"
    email = "applicationCommon::Email"
    edges = [
        {"id": "edge-1", "source_id": account, "target_id": contact, "attribute": "primaryContact",
         "source_uri": "https://example.org/account"},
        {"id": "edge-2", "source_id": email, "target_id": account, "attribute": "Sender_emailSender",
         "source_uri": "https://example.org/email"},
    ]

    class GraphIndex:
        graph = SimpleNamespace(entities={account: {"name": "Account"}, contact: {"name": "Contact"},
                                  email: {"name": "Email"}}, adjacency={account: edges})

        def search(self, *_args, **_kwargs):
            return []

    models = FakeModels()
    result = _answer_from_query(
        "List relationships declared by applicationCommon::Account to Contact and Email.",
        "query", GraphIndex(), models, rerank_enabled=True, limit=6, mode="hybrid")
    assert result["generator"] == "verified-graph-response"
    assert f"Outgoing: {account} --primaryContact--> {contact}" in result["answer"]
    assert f"Incoming: {email} --Sender_emailSender--> {account}" in result["answer"]
    assert models.calls == []


def test_query_is_rewritten_and_only_the_selected_passage_is_sent_to_generation(monkeypatch):
    monkeypatch.setenv("CDM_RAG_MIN_SIMILARITY", "0.6")
    models = FakeModels(outputs=[{"search": True, "query": "Account attributes"}],
                        answer="Account has an identifier.")
    index = FakeIndex()
    result = answer_question("Which fields does Account have?", index, models)
    assert index.search_calls[0][0] == "Account attributes\nWhich fields does Account have?"
    assert result["context"][0]["text"] == "Account attributes"
    assert "direct_neighbors" not in result["context"][0]
    assert result["sources"][0]["uri"] == "https://example.org/a"
    assert result["sources"][0]["display_citation"] == "[M::Account]"
    assert "Account has an identifier. [M::Account]" in result["answer"]
    assert "[R1]" not in result["answer"]
    assert result["retrieval"]["search_query"] == "Account attributes\nWhich fields does Account have?"
    assert result["answer_quality"]["status"] == "accepted"


def test_judge_receives_only_cited_evidence():
    sources = [{"citation": "R1", "entity_id": "M::Account", "kind": "entity", "text": "Account definition"},
               {"citation": "R2", "entity_id": "M::Private", "kind": "entity", "text": "Uncited definition"}]
    contexts = [{"citation": "R1", "entity_id": "M::Account", "name": "Account", "model": "M",
                 "description": "A bank account.", "attributes": [{"name": "accountNumber"}]},
                {"citation": "R2", "entity_id": "M::Private", "name": "Private", "model": "M",
                 "description": "Must not be sent.", "attributes": []}]
    models = FakeModels()
    assert review_answer("What is Account?", "A bank account. [R1]", sources, contexts, models)["decision"] == "accept"
    judge_message = models.messages[-1][1][-1]["content"]
    assert '"entity_id":"M::Account"' in judge_message
    assert "Must not be sent" not in judge_message


def test_judge_receives_directed_edge_endpoints_to_check_relationship_citations():
    source = {"citation": "R1", "entity_id": "M::Email", "kind": "entity", "text": "Email definition"}
    context = {"citation": "R1", "entity_id": "M::Email", "name": "Email", "model": "M",
               "description": "Email message", "attributes": [],
               "direct_neighbors": [{"citation": "R1", "source_entity_id": "M::Email",
                                    "target_entity_id": "M::Account", "attribute": "senderAccount",
                                    "entity_name": "Account"}]}
    models = FakeModels()
    review_answer("How does Account relate to Email?", "Account --senderAccount--> Email [R1]",
                  [source], [context], models)
    judge_message = models.messages[-1][1][-1]["content"]
    assert '"source_entity_id":"M::Email"' in judge_message
    assert '"target_entity_id":"M::Account"' in judge_message


def test_judge_does_not_promote_an_unacceptable_verdict_to_accept():
    contradictory = {"decision": "retry_answer", "issue": "unsupported_claim",
                     "feedback": "The source does not support this.", "search_query": "",
                     "claim_checks": [{"claim": "The answer is supported.", "assessment": "supported"}],
                     "coverage": "answered", "verdict": "unacceptable"}
    models = FakeModels(judge_decisions=[contradictory])
    result = review_answer("What is Account?", "A bank account. [R1]",
                           [{"citation": "R1", "entity_id": "M::Account", "kind": "entity", "text": "Account definition"}],
                           [{"citation": "R1", "entity_id": "M::Account", "name": "Account", "model": "M",
                             "description": "A bank account.", "attributes": []}], models)
    assert result["decision"] == "retry_answer"
    assert result["reconciled"] is False


def test_accept_decision_with_uncertain_verdict_abstains():
    contradictory = {**accepted_judge(), "verdict": "uncertain"}
    result = review_answer("What is Account?", "An account. [R1]",
                           [{"citation": "R1", "entity_id": "M::Account", "kind": "entity", "text": "Account definition"}],
                           [], FakeModels(judge_decisions=[contradictory]))
    assert result["decision"] == "abstain"


def test_judge_feedback_retries_answer_once_and_accepts_the_correction(monkeypatch):
    monkeypatch.setenv("CDM_RAG_MIN_SIMILARITY", "0.6")
    models = FakeModels(outputs=[{"search": True, "query": "Account definition"}, {"ranked_ids": ["C1"]}],
                        judge_decisions=[answer_retry(), accepted_judge()],
                        answer_claims=[
                            [{"text": "Account is a data entity.", "citations": ["R1"]}],
                            [{"text": "Account has an identifier attribute.", "citations": ["R1"]}],
                        ])
    index = FakeIndex()
    result = answer_question("Describe Account", index, models)
    assert result["answer_quality"]["status"] == "accepted_after_retry"
    assert result["answer_quality"]["answer_attempts"] == 2
    assert result["answer_quality"]["retrieval_retries"] == 0
    assert "identifier attribute" in result["answer"]
    assert len(index.search_calls) == 1
    answer_call = [messages for model, messages in models.messages if model == models.answer_model][-1]
    assert "Name the identifier attribute." in answer_call[-1]["content"]


def test_judge_feedback_can_retrieve_again_once(monkeypatch):
    monkeypatch.setenv("CDM_RAG_MIN_SIMILARITY", "0.6")
    models = FakeModels(outputs=[{"search": True, "query": "Account"},
                                 {"ranked_ids": ["C1"]}, {"ranked_ids": ["C1"]}],
                        judge_decisions=[retrieval_retry(), accepted_judge()],
                        answer_claims=[
                            [{"text": "Account is a data entity.", "citations": ["R1"]}],
                            [{"text": "Account has an identifier attribute.", "citations": ["R1"]}],
                        ])
    index = FakeIndex()
    result = answer_question("Describe Account", index, models)
    assert result["answer_quality"]["status"] == "accepted_after_retry"
    assert result["answer_quality"]["retrieval_retries"] == 1
    assert len(index.search_calls) == 2
    assert "Account identifier attribute" in index.search_calls[1][0]
    assert result["retrieval"]["search_query"] == "Account\nDescribe Account Account identifier attribute"


def test_judge_rejects_second_attempt_and_returns_a_cited_answer_free_abstention(monkeypatch):
    monkeypatch.setenv("CDM_RAG_MIN_SIMILARITY", "0.6")
    models = FakeModels(outputs=[{"search": True, "query": "Account"}, {"ranked_ids": ["C1"]}],
                        judge_decisions=[answer_retry(), answer_retry("Still missing the requested detail.")])
    result = answer_question("Describe Account", FakeIndex(), models)
    assert result["answer_quality"]["status"] == "abstained_after_retry_rejected"
    assert result["answer_quality"]["answer_attempts"] == 2
    assert result["route"] == "no_evidence"
    assert result["sources"] == []
    assert "couldn't verify" in result["answer"]


def test_judge_failure_does_not_stream_or_return_an_unreviewed_answer(monkeypatch):
    monkeypatch.setenv("CDM_RAG_MIN_SIMILARITY", "0.6")
    models = FakeModels(outputs=[{"search": True, "query": "Account"}, {"ranked_ids": ["C1"]}],
                        judge_decisions=[ModelError("judge offline")])
    streamed = []
    result = answer_question("Describe Account", FakeIndex(), models, on_token=streamed.append)
    assert result["answer_quality"]["status"] == "unavailable"
    assert result["answer_quality"]["judge_calls"] == 1
    assert result["sources"] == []
    assert "couldn't verify" in result["answer"]
    assert streamed == []


def test_invalid_labels_are_never_accepted_as_citations():
    models = FakeModels(claims=[{"text": "Invented claim.", "citations": ["R99"]}])
    sources = [{"citation": "R1", "text": "Source content"}]
    answer, generator = cited_answer("Question?", sources, [{"citation": "R1"}], models)
    assert generator == "extractive-fallback"
    assert "Invented claim" not in answer and "[R1]" in answer


def test_answer_prompt_requires_explicit_relationships_and_missing_parts():
    models = FakeModels(answer="The evidence is incomplete.")
    cited_answer("How are these connected?", [{"citation": "R1", "text": "A definition"}],
                 [{"citation": "R1", "text": "A definition"}], models)
    prompt = models.messages[-1][1][0]["content"]
    assert "explicitly names its source, attribute, and target" in prompt
    assert "preserve all stated alternatives" in prompt
    assert "identify the requested parts for which evidence is missing" in prompt


def test_separate_generated_claims_keep_their_own_supporting_entities():
    sources = [{"citation": "R1", "entity_id": "M::Account", "text": "Account definition"},
               {"citation": "R2", "entity_id": "M::Contact", "text": "Contact definition"}]
    models = FakeModels(claims=[
        {"text": "An account has an identifier.", "citations": ["R1"]},
        {"text": "A contact has a name.", "citations": ["R2"]},
    ])
    raw, generator = cited_answer("What are they?", sources, [], models)
    answer, cited = format_entity_citations(raw, sources)
    assert answer == ("An account has an identifier. [M::Account] "
                      "A contact has a name. [M::Contact]")
    assert [source["entity_id"] for source in cited] == ["M::Account", "M::Contact"]
    assert generator == "answer"


@pytest.mark.parametrize("claims", [
    [{"text": "Unsupported claim.", "citations": []}],
    [{"text": "Unsupported claim [R99].", "citations": ["R1"]}],
])
def test_generation_without_valid_source_labels_uses_extracts(claims):
    answer, generator = cited_answer("Question?", [{"citation": "R1", "text": "Source content"}],
                                     [{"citation": "R1"}], FakeModels(claims=claims))
    assert generator == "extractive-fallback"
    assert "Unsupported claim" not in answer
    assert "Source content [R1]" in answer


def test_short_model_answer_is_returned_with_its_citation_without_expansion():
    models = FakeModels(answer="FHLineOfCredit")
    sources = [{"citation": "R1", "text": "A borrowing limit"}]
    contexts = [{"citation": "R1", "text": "A borrowing limit"}]
    answer, generator = cited_answer("Which entity?", sources, contexts, models)
    assert answer == "FHLineOfCredit [R1]"
    assert generator == "answer"


def test_generation_passage_limit_keeps_only_top_passages_and_reports_omissions():
    candidates = [
        {"id": "p1", "entity_id": "M::Account", "kind": "attributes", "text": "Account id",
         "uri": "https://example.org/a"},
        {"id": "p2", "entity_id": "M::Contact", "kind": "entity", "text": "Contact definition",
         "uri": "https://example.org/c"},
    ]
    contexts, sources, omitted = prepare_passages(candidates, 1)
    assert [context["passage_id"] for context in contexts] == ["p1"]
    assert [source["entity_id"] for source in sources] == ["M::Account"]
    assert omitted == ["p2"]


def test_source_labels_distinguish_passages_even_when_entities_share_a_source_file():
    contexts, sources, _ = prepare_passages([
        {"id": "a", "entity_id": "M::A", "kind": "entity", "text": "A", "uri": "https://example.org/shared"},
        {"id": "b", "entity_id": "M::B", "kind": "entity", "text": "B", "uri": "https://example.org/shared"},
    ], 2)
    assert [context["citation"] for context in contexts] == ["R1", "R2"]
    assert [source["entity_id"] for source in sources] == ["M::A", "M::B"]


def test_inline_citations_use_qualified_entity_names_and_keep_sources_distinct():
    sources = [
        {"citation": "R1", "entity_id": "banking::Account", "uri": "https://example.org/a"},
        {"citation": "R2", "entity_id": "common::Account", "uri": "https://example.org/b"},
        {"citation": "R3", "entity_id": "banking::Account", "uri": "https://example.org/c"},
        {"citation": "R4", "entity_id": "M::Uncited", "uri": "https://example.org/d"},
    ]
    answer, cited = format_entity_citations("First [R1, R2], another source [R3].", sources)
    assert answer == ("First [banking::Account · R1] [common::Account], "
                      "another source [banking::Account · R3].")
    assert [source["display_citation"] for source in cited] == [
        "[banking::Account · R1]", "[common::Account]", "[banking::Account · R3]"]
    assert len(cited) == 3
    with pytest.raises(ModelError, match="unavailable source"):
        format_entity_citations("Bad reference [R99].", sources)


def test_reranker_rejects_ids_not_in_candidates():
    models = FakeModels(outputs=[{"ranked_ids": ["C999"]}])
    with pytest.raises(ModelError, match="invalid passage IDs"):
        rerank("Account?", [{"text": "Account"}], models, 6)


def test_invalid_query_plan_fails_closed():
    models = FakeModels(outputs=[{"search": True, "query": ""}])
    with pytest.raises(ModelError, match="did not provide"):
        plan_search("Account?", models)


def test_catalog_render_keeps_entity_identity_and_direct_relationships():
    catalog = {"entities": [{"id": "M::A", "name": "A", "model": "M", "source_uri": "https://x/A",
                              "attributes": [{"name": "id", "data_type": "string", "description": "Identifier"}],
                              "relationships": [{"attribute": "owner", "target_ids": ["M::B"]}]}]}
    chunks = render_chunks(catalog)
    entity_chunks = [item for item in chunks if item["kind"] != "catalog"]
    assert {item["kind"] for item in entity_chunks} == {"entity", "attributes", "relationship"}
    assert all(item["entity_id"] == "M::A" and item["uri"] == "https://x/A" for item in entity_chunks)
    assert "M::B" in next(item["text"] for item in chunks if item["kind"] == "relationship")


def test_no_direct_relationship_reports_a_verifiable_path_without_claiming_a_direct_link():
    class RelationIndex:
        def relationship_evidence(self, _first, _second):
            return {"direct": [], "path": [
                {"source_id": "M::Contact", "target_id": "M::User", "attribute": "preferredUser",
                 "source_uri": "https://example.org/contact"},
                {"source_id": "M::User", "target_id": "M::Organization", "attribute": "organization",
                 "source_uri": "https://example.org/user"},
            ]}

    answer, sources = no_direct_relationship_answer(RelationIndex(), "M::Contact", "M::Organization", [])
    assert "No direct relationship was extracted" in answer
    assert "M::Contact --preferredUser--> M::User --organization--> M::Organization" in answer
    assert "not proof of a direct business relationship" in answer
    assert {source["uri"] for source in sources} == {"https://example.org/contact", "https://example.org/user"}


def test_relationship_answer_does_not_reverse_an_extracted_edge():
    class ReverseOnlyIndex:
        def relationship_evidence(self, _first, _second):
            return {"direct": [{"source_id": "M::Account", "target_id": "M::Contact",
                                "attribute": "primaryContact", "source_uri": "https://example.org/account"}],
                    "path": []}

    answer, sources = no_direct_relationship_answer(ReverseOnlyIndex(), "M::Contact", "M::Account", [])
    assert "M::Account --primaryContact--> M::Contact" in answer
    assert "No extracted direct edge points from M::Contact to M::Account" in answer
    assert "[R1]" in answer
    assert sources[0]["entity_id"] == "M::Account"


def test_relationship_answer_reports_each_direct_direction_with_its_source():
    class BidirectionalIndex:
        def relationship_evidence(self, _first, _second):
            return {"direct": [
                {"source_id": "M::Contact", "target_id": "M::Account", "attribute": "parentCustomer",
                 "source_uri": "https://example.org/contact"},
                {"source_id": "M::Account", "target_id": "M::Contact", "attribute": "primaryContact",
                 "source_uri": "https://example.org/account"}], "path": []}

    answer, sources = no_direct_relationship_answer(BidirectionalIndex(), "M::Contact", "M::Account", [])
    assert "M::Contact --parentCustomer--> M::Account" in answer
    assert "M::Account --primaryContact--> M::Contact" in answer
    assert {source["entity_id"] for source in sources} == {"M::Contact", "M::Account"}


def test_path_citations_do_not_merge_different_entities_in_the_same_source_file():
    class SharedFilePath:
        def relationship_evidence(self, _first, _second):
            return {"direct": [], "path": [
                {"source_id": "M::Contact", "target_id": "M::User", "attribute": "preferredUser",
                 "source_uri": "https://example.org/shared"},
                {"source_id": "M::User", "target_id": "M::Organization", "attribute": "organization",
                 "source_uri": "https://example.org/shared"},
            ]}

    answer, sources = no_direct_relationship_answer(SharedFilePath(), "M::Contact", "M::Organization", [])
    named_answer, named_sources = format_entity_citations(answer, sources)
    assert "[M::Contact]" in named_answer and "[M::User]" in named_answer
    assert {source["entity_id"] for source in named_sources} == {"M::Contact", "M::User"}


@pytest.mark.parametrize('category,fragment', [
    ('records', 'actual customer or account records'),
    ('runtime', 'deployed application or workflow'),
    ('mixed', 'ask the CDM question separately'),
    ('unrelated', 'Microsoft CDM'),
    ('unclear', 'clarify'),
])
def test_scope_guard_declines_before_planning_and_retrieval(category, fragment):
    models, index = FakeModels(scope=category), FakeIndex(similarity=0.9)
    result = answer_question('A potentially unsupported question', index, models)
    assert result['route'] == 'no_search' and fragment in result['answer']
    assert result['sources'] == [] and result['context'] == []
    assert result['retrieval']['scope_category'] == category
    assert index.search_calls == [] and models.calls == ['scope']


def test_scope_model_error_fails_closed_without_exposing_sources():
    models, index = FakeModels(scope=ModelError('offline')), FakeIndex(similarity=0.9)
    result = answer_question('Some question', index, models)
    assert result['route'] == 'no_search' and result['retrieval']['scope_category'] == 'unavailable'
    assert result['sources'] == [] and index.search_calls == [] and models.calls == ['scope']
    assert 'offline' not in result['answer']


def test_invalid_scope_category_fails_closed():
    models, index = FakeModels(scope='unexpected'), FakeIndex(similarity=0.9)
    result = answer_question('Some question', index, models)
    assert result['route'] == 'no_search' and result['retrieval']['scope_category'] == 'unavailable'
    assert result['sources'] == [] and index.search_calls == []
