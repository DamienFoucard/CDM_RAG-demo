"""The API describes actual model calls and short circuits honestly."""
import asyncio
import json
import threading

import pytest

from app import main
from app.runtime.query_trace import TracedModels, active_trace, trace_embedding
from app.runtime.ollama import Ollama


class Models:
    embedding_model = 'embed:1'
    router_model = 'query:1'
    decision_model = 'decision:1'
    scope_model = 'scope:1'
    judge_model = 'judge:1'
    answer_model = 'answer:1'

    @trace_embedding
    def embed(self, texts, *, query):
        return [[1.0]]

    def chat_json_timed(self, model, *args, **kwargs):
        if model == 'failed:1':
            raise RuntimeError('model failed')
        return {}, None

    def chat_json(self, model, *args, **kwargs):
        return {}

    def chat_stream(self, model, *args, **kwargs):
        return 'answer'

    def unload_other_models(self, model):
        pass


def setup_query(monkeypatch):
    monkeypatch.setattr(main.app.state, 'models', Models(), raising=False)
    monkeypatch.setattr(main.app.state, 'index', object(), raising=False)
    monkeypatch.setattr(main.app.state, 'query_router', None, raising=False)
    monkeypatch.setattr(main.app.state, 'judge_enabled', False, raising=False)
    monkeypatch.setattr(main.app.state, 'model_run_lock', threading.Lock(), raising=False)
    monkeypatch.setattr('app.runtime.observability._write_event', lambda *args, **kwargs: None)


def test_no_search_has_only_embedding_and_queue_timing(monkeypatch):
    setup_query(monkeypatch)
    def answer(question, index, models, **kwargs):
        kwargs['progress']('checking_relevance')
        models.embed(['Hi'], query=True)
        kwargs['progress']('no_schema_search')
        return {'answer': 'No search', 'route': 'no_search', 'retrieval': {}, 'sources': []}
    monkeypatch.setattr(main, 'answer_question', answer)
    stages, model_events = [], []
    result = main._run_logged_query(main.Question(question='Hi'), 'test', on_stage=stages.append,
                                    on_model=model_events.append)
    info = result['processing']
    assert stages[0] == 'waiting_for_model'
    assert info['queue_ms'] >= 0 and info['elapsed_ms'] >= info['queue_ms']
    assert [x['role'] for x in info['model_calls']] == ['embedding']
    assert info['answer_method'] == 'no_generation'
    assert model_events[0]['model'] == 'embed:1'
    assert all(x['status'] == 'completed' for x in info['stage_timings'])


def test_verified_answer_does_not_claim_answer_model_was_called(monkeypatch):
    setup_query(monkeypatch)
    def answer(question, index, models, **kwargs):
        p = kwargs['progress']
        p('checking_relevance'); models.embed(['What?'], query=True)
        p('checking_scope'); models.chat_json_timed('scope:1', [])
        p('planning_search'); models.chat_json_timed('decision:1', []); models.chat_json('query:1', [])
        p('searching_catalog'); models.embed(['rewritten'], query=True); p('checking_citations')
        return {'answer': 'Verified', 'route': 'cdm', 'sources': [],
                'retrieval': {'generator': 'verified-graph-response', 'router_backend': 'ollama_evidence_decision'}}
    monkeypatch.setattr(main, 'answer_question', answer)
    result = main._run_logged_query(main.Question(question='What?'), 'test')
    calls = result['processing']['model_calls']
    assert [(x['role'], x['model']) for x in calls] == [
        ('embedding', 'embed:1'), ('scope', 'scope:1'), ('routing', 'decision:1'), ('query', 'query:1'),
        ('embedding', 'embed:1')]
    assert result['processing']['answer_method'] == 'verified-graph-response'
    assert result['processing']['configured_models']['answer'] == 'answer:1'


def test_failed_model_call_is_marked_failed():
    tracer = TracedModels(Models(), lambda: 'checking_scope')
    with pytest.raises(RuntimeError):
        tracer.chat_json_timed('failed:1', [])
    assert tracer.calls == [{'role': 'scope', 'model': 'failed:1',
                             'stage': 'checking_scope', 'outcome': 'failed'}]


def test_shared_index_client_traces_only_the_current_request(monkeypatch):
    client = Ollama('http://example.test', 'qwen3-embedding:0.6b', 'query', 'answer')
    monkeypatch.setattr(client, '_post', lambda *a, **k: {'embeddings': [[1.0]]})
    trace = TracedModels(client, lambda: 'searching_catalog')
    marker = active_trace.set(trace)
    try:
        client.embed(['question'], query=True)
    finally:
        active_trace.reset(marker)
    client.embed(['outside request'], query=True)
    assert len(trace.calls) == 1
    assert trace.calls[0]['stage'] == 'searching_catalog'
    assert trace.calls[0]['model'] == 'qwen3-embedding:0.6b'


def test_stream_starts_with_waiting_and_final_result(monkeypatch):
    def run(body, query_id, *, on_stage, on_token, on_model, request_started_at):
        on_stage('checking_scope')
        on_model({'role': 'scope', 'model': 'scope:1', 'stage': 'checking_scope', 'outcome': 'started'})
        on_token('draft')
        return {'answer': 'final', 'processing': {'elapsed_ms': 1}}
    monkeypatch.setattr(main, '_run_logged_query', run)
    async def collect():
        response = await main.query_stream(main.Question(question='What?'), main.Request({'type': 'http', 'state': {}}))
        return [chunk async for chunk in response.body_iterator]
    chunks = asyncio.run(collect())
    events = [json.loads(chunk[6:]) for chunk in chunks if chunk.startswith('data: ')]
    assert events[0]['stage'] == 'waiting_for_model'
    assert any(e.get('type') == 'model' and e['model'] == 'scope:1' for e in events)
    assert any(e.get('type') == 'token' and e['provisional'] is True for e in events)
    assert events[-1]['result']['answer'] == 'final'
