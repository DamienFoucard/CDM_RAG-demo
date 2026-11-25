from app.retrieval.query_anchor import anchor_search_query
from app.retrieval.rag import plan_search_timed, rewrite_search_query


def test_contextual_rewrite_keeps_user_details_and_excludes_assistant_claim():
    messages = [
        {"role": "user", "content": "A business unit relates to an individual such as a customer, supplier, or colleague."},
        {"role": "assistant", "content": "This is definitely the fictional entity InventedPerson."},
    ]
    query = anchor_search_query("person entity in applicationCommon", "Which entity models that person?", messages)
    assert "Which entity models that person?" in query
    assert "customer, supplier, or colleague" in query
    assert "InventedPerson" not in query


def test_unrelated_previous_turn_is_not_appended_for_a_new_question():
    query = anchor_search_query("loan holding", "Find the loan holding definition",
                                [{"role": "user", "content": "Show me an overdraft."}])
    assert "Find the loan holding definition" in query
    assert "overdraft" not in query


def test_anchor_does_not_duplicate_an_identical_question():
    assert anchor_search_query("What is Account?", "What is Account?") == "What is Account?"


def test_rewrite_prompt_retains_requested_relationships_without_guessing():
    captured = []
    class Model:
        router_model = "router"

        def chat_json(self, model, messages, **kwargs):
            captured.extend(messages)
            return {"query": "financial holding customer"}

    original = "How does the card link to the holding and identify its customer?"
    query = rewrite_search_query(original, Model())
    assert original in query
    assert "Do not assume that two entities are related" in captured[0]["content"]
    assert "add unrelated schema categories" in captured[0]["content"]


def test_planner_retains_followup_and_prior_user_context():
    class Model:
        router_model = "router"

        def chat_json_timed(self, model, messages):
            return {"search": True, "query": "person record"}, {"total_ms": 1}

    context = [{"role": "user", "content": "A business unit relates to a customer or supplier."},
               {"role": "assistant", "content": "AssumedExample"}]
    search, query, _ = plan_search_timed("Which entity models that person?", Model(), context)
    assert search
    assert "customer or supplier" in query
    assert "Which entity models that person?" in query
    assert "AssumedExample" not in query


def test_planner_never_appends_context_when_it_declines_search():
    class Model:
        router_model = "router"

        def chat_json_timed(self, model, messages):
            return {"search": False, "query": ""}, None

    search, query, _ = plan_search_timed("What is that?", Model(), [{"role": "user", "content": "hi"}])
    assert not search and query == ""
