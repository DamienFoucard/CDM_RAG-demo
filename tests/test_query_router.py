from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from app.runtime.ollama import ModelError
from app.retrieval.decision_router import LocalDecisionRouter, evidence_messages, initial_hits
from app.retrieval.rag import answer_question, plan_routed_search
from app.retrieval.router import JevRouter, RouteDecision


class Planner:
    router_model = 'local'

    def __init__(self, result=None):
        self.result = result or {'search': True, 'query': 'local rewrite'}
        self.calls = 0

    def chat_json(self, *args, **kwargs):
        self.calls += 1
        return self.result

    def chat_json_timed(self, *args, **kwargs):
        return self.chat_json(*args, **kwargs), None


def decision(choice='raw_search', confidence=.9):
    return RouteDecision(choice, confidence, {choice: confidence}, 'jev-test', 20, 2)


def test_jev_request_and_typed_response():
    def handler(request):
        assert request.url.path == '/v1/systemone'
        assert request.headers['Authorization'] == 'Bearer test-key'
        data = json.loads(request.content)
        assert data['state'] == 'Which contact relates to an account?'
        assert set(data['questions']['route']['criteria']) == {'raw_search', 'rewrite_search', 'no_search'}
        return httpx.Response(200, json={'model':'jev-test','answers':{'route':{
            'type':'choice','choice':'raw_search','confidence':.9,
            'probabilities':{'raw_search':.9,'rewrite_search':.05,'no_search':.05}}},
            'usage':{'input_tokens':20,'output_tokens':2}})
    router = JevRouter('test-key')
    router.client.close()
    router.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        answer = router.classify('Which contact relates to an account?')
        assert (answer.choice, answer.confidence, answer.input_tokens) == ('raw_search', .9, 20)
    finally:
        router.close()


@pytest.mark.parametrize('status,body', [(429, {}), (200, {'answers': {}}),
    (200, {'answers': {'route': {'type':'choice','choice':'no_search','confidence':float('nan'),
                                'probabilities':{}}}, 'usage':{'input_tokens':1,'output_tokens':1}})])
def test_bad_hosted_response_is_rejected(status, body):
    router = JevRouter('test-key')
    router.client.close()
    router.client = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(status, json=body)))
    try:
        with pytest.raises(ModelError):
            router.classify('question')
    finally:
        router.close()


def test_high_confidence_raw_query_avoids_local_planner():
    planner = Planner()
    router = SimpleNamespace(classify=lambda _: decision(), min_confidence=.7, model='jev-test')
    search, query, diag = plan_routed_search('original question', planner, router)
    assert search and query == 'original question' and planner.calls == 0
    assert diag['router_decision'] == 'raw_search' and not diag['router_fallback_used']


def test_low_confidence_falls_back_to_local_planner():
    planner = Planner()
    router = SimpleNamespace(classify=lambda _: decision('no_search', .5), min_confidence=.7, model='jev-test')
    search, query, diag = plan_routed_search('original question', planner, router)
    assert search and query == 'local rewrite\noriginal question' and planner.calls == 1
    assert diag['router_fallback_used'] and diag['router_decision'] == 'confidence_fallback_search'


def test_service_failure_falls_back_to_local_planner():
    def fail(_question):
        raise ModelError('unavailable')
    router = SimpleNamespace(classify=fail, min_confidence=.7, model='jev-test')
    planner = Planner({'search':False,'query':''})
    search, query, diag = plan_routed_search('question', planner, router)
    assert not search and query == '' and planner.calls == 1
    assert diag['router_decision'] == 'error_fallback_no_search' and diag['router_fallback_used']


def test_high_confidence_no_search_avoids_local_planner():
    planner = Planner()
    router = SimpleNamespace(classify=lambda _: decision('no_search'), min_confidence=.7, model='jev-test')
    search, query, diag = plan_routed_search('hello', planner, router)
    assert not search and query == '' and planner.calls == 0
    assert diag['router_decision'] == 'no_search'


def test_high_confidence_rewrite_uses_local_rewriter():
    planner = Planner({'query':'customer account definitions'})
    router = SimpleNamespace(classify=lambda _: decision('rewrite_search'), min_confidence=.7, model='jev-test')
    search, query, diag = plan_routed_search('question', planner, router)
    assert search and query == 'customer account definitions\nquestion' and planner.calls == 1
    assert diag['router_decision'] == 'rewrite_search' and diag['query_generated']


class SplitLocalModels:
    """Decision and rewrite requests must target different explicit tags."""
    router_model = 'rewriter-9b'
    embedding_model = 'nomic-embed-text'
    scope_model = 'scope'

    def __init__(self, route):
        self.route = route
        self.calls = []
        self.messages = []

    def chat_json_timed(self, model, messages, **kwargs):
        self.calls.append(('decision', model))
        self.messages.append(messages)
        if isinstance(self.route, Exception):
            raise self.route
        return {'route': self.route}, {'eval_count': 3}

    def chat_json(self, model, messages, **kwargs):
        self.calls.append(('rewrite', model))
        return {'query': 'rewritten schema'}


@pytest.mark.parametrize('route,expected', [
    ('raw_search', (True, 'original question')),
    ('no_search', (False, '')),
    ('rewrite_search', (True, 'rewritten schema\noriginal question')),
])
def test_local_decision_and_query_models_are_independent(route, expected):
    models = SplitLocalModels(route)
    search, query, diag = plan_routed_search('original question', models, LocalDecisionRouter('decision-4b'))
    assert (search, query) == expected
    assert diag['router_model'] == 'decision-4b' and diag['router_decision'] == route
    assert models.router_model == 'rewriter-9b'  # no mutation of the shared client
    assert models.calls == ([('decision', 'decision-4b'), ('rewrite', 'rewriter-9b')]
                            if route == 'rewrite_search' else [('decision', 'decision-4b')])


def test_local_decision_error_fails_closed_and_returns_retry_message(monkeypatch):
    models = SplitLocalModels(ModelError('internal details'))
    monkeypatch.setattr('app.retrieval.rag.assess_scope', lambda question, client: 'schema')
    index = SimpleNamespace(semantic_probe=lambda question: 1.0)
    result = answer_question('original question', index, models, router=LocalDecisionRouter('decision-4b'),
                             judge_enabled=False)
    assert result['route'] == 'no_search'
    assert result['retrieval']['router_decision'] == 'error_no_search'
    assert 'try again' in result['answer'].lower() and 'internal details' not in result['answer']
    assert models.calls == [('decision', 'decision-4b')]


def test_local_router_does_not_bypass_existing_scope_gate(monkeypatch):
    models = SplitLocalModels('raw_search')
    monkeypatch.setattr('app.retrieval.rag.assess_scope', lambda question, client: 'records')
    index = SimpleNamespace(semantic_probe=lambda question: 1.0)
    result = answer_question('my account balance', index, models, router=LocalDecisionRouter('decision-4b'),
                             judge_enabled=False)
    assert result['route'] == 'no_search' and result['retrieval']['router_decision'] == 'scope_records'
    assert not models.calls


def test_local_router_rewrite_error_does_not_invent_query():
    models = SplitLocalModels('rewrite_search')
    def fail(*args, **kwargs):
        raise ModelError('rewrite unavailable')
    models.chat_json = fail
    with pytest.raises(ModelError, match='rewrite unavailable'):
        plan_routed_search('original question', models, LocalDecisionRouter('decision-4b'))


def test_evidence_router_uses_identical_bounded_benchmark_prompt_without_reranking():
    events = []
    passages = [{'id': f'p{i}', 'title': 'title', 'text': 'x' * 900, 'score': .1} for i in range(4)]

    class Index:
        def search(self, query, *, limit, mode):
            events.append(('search', query, limit, mode))
            return passages

    models = SplitLocalModels('raw_search')
    search, query, diag = plan_routed_search('original question', models,
                                           LocalDecisionRouter('decision-4b', use_evidence=True), index=Index())
    assert search and query == 'original question'
    assert events == [('search', 'original question', 3, 'hybrid')]
    assert models.calls == [('decision', 'decision-4b')]
    assert models.messages[0] == evidence_messages({'question': 'original question'}, initial_hits(passages))
    assert diag['router_backend'] == 'ollama_evidence_decision' and diag['router_presearch_count'] == 3
    assert all('x' * 701 not in m['content'] for m in models.messages[0])
    assert 'p0' not in models.messages[0][0]['content']  # untrusted evidence is never system text
    assert 'p0' in models.messages[0][-1]['content']


def test_evidence_presearch_failure_fails_closed_before_classifier(monkeypatch):
    models = SplitLocalModels('raw_search')
    monkeypatch.setattr('app.retrieval.rag.assess_scope', lambda question, client: 'schema')
    def fail(*args, **kwargs):
        raise RuntimeError('private storage details')
    index = SimpleNamespace(semantic_probe=lambda question: 1.0, search=fail)
    result = answer_question('schema question', index, models,
                             router=LocalDecisionRouter('decision-4b', use_evidence=True), judge_enabled=False)
    assert result['route'] == 'no_search' and result['retrieval']['router_decision'] == 'error_no_search'
    assert 'try again' in result['answer'].lower() and 'private storage details' not in result['answer']
    assert models.calls == []


@pytest.mark.parametrize('route,expected', [
    ('raw_search', (True, 'original question')),
    ('no_search', (False, '')),
    ('rewrite_search', (True, 'rewritten schema\noriginal question')),
])
def test_evidence_router_respects_all_three_routes_with_separate_query_model(route, expected):
    models = SplitLocalModels(route)
    index = SimpleNamespace(search=lambda query, limit, mode: [{'id': 'p1', 'text': 'catalog data'}])
    search, query, diag = plan_routed_search('original question', models,
                                           LocalDecisionRouter('decision-4b', use_evidence=True), index=index)
    assert (search, query) == expected
    assert diag['router_decision'] == route and models.router_model == 'rewriter-9b'
    assert models.calls == ([('decision', 'decision-4b'), ('rewrite', 'rewriter-9b')]
                            if route == 'rewrite_search' else [('decision', 'decision-4b')])


def test_evidence_router_runs_only_after_semantic_and_scope_gates(monkeypatch):
    models = SplitLocalModels('raw_search')
    events = []
    def scope(question, client):
        events.append('scope')
        return 'records'
    monkeypatch.setattr('app.retrieval.rag.assess_scope', scope)
    index = SimpleNamespace(semantic_probe=lambda question: events.append('semantic') or 1.0,
                            search=lambda *a, **k: events.append('search'))
    result = answer_question('my account balance', index, models,
                             router=LocalDecisionRouter('decision-4b', use_evidence=True), judge_enabled=False)
    assert result['route'] == 'no_search' and events == ['semantic', 'scope'] and models.calls == []
