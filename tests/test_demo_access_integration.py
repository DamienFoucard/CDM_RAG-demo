"""Exercise the demo boundary through actual HTTP routes without model calls."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.security import api_tokens
from app import main


@pytest.fixture
def demo_app(tmp_path, monkeypatch):
    registry = api_tokens.TokenRegistry(tmp_path / 'tokens.sqlite3')
    demo = registry.create('external demo', 'demo')
    owner = registry.create('owner client')
    fake_calls = []

    def answer(body, query_id, **callbacks):
        fake_calls.append(body)
        if 'on_stage' in callbacks:
            callbacks['on_stage']('checking_scope')
            callbacks['on_model']({'role': 'scope', 'model': 'scope:test',
                                   'stage': 'checking_scope', 'outcome': 'started'})
            callbacks['on_token']('provisional')
        return {'query_id': query_id, 'answer': 'A grounded answer [R1].',
                'sources': [{'citation': 'R1'}],
                'processing': {'answer_method': 'test'}}

    monkeypatch.setattr(main, '_run_logged_query', answer)
    app = FastAPI()
    app.add_middleware(api_tokens.APIAuthMiddleware, required=True, registry=registry,
                       proxy_secret='a' * 64, owner_email='owner@example.test',
                       owner_subject='owner-sub',
                       demo_expires_at=(datetime.now(timezone.utc) + timedelta(days=7)).isoformat())
    app.add_api_route('/api/v1/query', main.query, methods=['POST'])
    app.add_api_route('/api/v1/query/stream', main.query_stream, methods=['POST'])
    with TestClient(app) as client:
        yield client, registry, demo, owner, fake_calls


def test_query_enforces_demo_options_at_http_boundary(demo_app):
    client, _, demo, owner, calls = demo_app
    demo_headers = {'Authorization': 'Bearer ' + demo['token']}
    owner_headers = {'Authorization': 'Bearer ' + owner['token']}

    assert client.post('/api/v1/query', json={'question': 'What is Account?'}).status_code == 401
    for override in ({'rerank': True}, {'limit': 12}, {'mode': 'lexical'}):
        request = {'question': 'What is Account?', **override}
        assert client.post('/api/v1/query', headers=demo_headers, json=request).status_code == 403
        assert calls == []
        assert client.post('/api/v1/query', headers=owner_headers, json=request).status_code == 200
        assert calls.pop().question == 'What is Account?'

    # A demo bearer must not gain owner privileges from accompanying headers.
    spoofed = {**demo_headers, 'X-CDM-RAG-Owner-Session': 'a' * 64,
               'X-CDM-RAG-Owner-Email': 'owner@example.test',
               'X-CDM-RAG-Owner-Subject': 'owner-sub'}
    assert client.post('/api/v1/query', headers=spoofed,
                       json={'question': 'What is Account?', 'rerank': True}).status_code == 403
    assert calls == []

    accepted = client.post('/api/v1/query', headers=demo_headers,
                           json={'question': '  What is\n Account?  '})
    assert accepted.status_code == 200
    assert accepted.headers['x-query-id'] == accepted.json()['query_id']
    assert accepted.json()['sources'] == [{'citation': 'R1'}]
    assert calls.pop().question == 'What is Account?'
    assert client.post('/api/v1/query', headers=demo_headers,
                       json={'question': ' ', 'unexpected': True}).status_code == 422
    assert calls == []


def test_streamed_demo_question_emits_progress_and_authoritative_result(demo_app):
    client, _, demo, _, calls = demo_app
    response = client.post('/api/v1/query/stream',
                           headers={'Authorization': 'Bearer ' + demo['token']},
                           json={'question': 'What is Account?'})
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/event-stream')
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert events[0]['stage'] == 'waiting_for_model'
    assert any(event.get('stage') == 'checking_scope' for event in events)
    assert any(event.get('type') == 'model' and event['model'] == 'scope:test' for event in events)
    assert any(event.get('type') == 'token' and event['provisional'] is True for event in events)
    assert events[-1]['result']['answer'] == 'A grounded answer [R1].'
    assert events[-1]['result']['processing']['answer_method'] == 'test'
    assert response.headers['x-query-id'] == events[-1]['result']['query_id']
    assert len(calls) == 1


def test_stream_failure_is_sanitized_and_does_not_block_next_demo_request(demo_app, monkeypatch):
    client, _, demo, _, _ = demo_app
    headers = {'Authorization': 'Bearer ' + demo['token']}
    attempts = []

    def answer(body, query_id, **callbacks):
        attempts.append(query_id)
        if len(attempts) == 1:
            raise RuntimeError('private internal failure details')
        return {'query_id': query_id, 'answer': 'Recovered'}

    monkeypatch.setattr(main, '_run_logged_query', answer)
    failed = client.post('/api/v1/query/stream', headers=headers, json={'question': 'First'})
    events = [json.loads(line[6:]) for line in failed.text.splitlines() if line.startswith('data: ')]
    assert events[-1]['type'] == 'error'
    assert 'private internal failure details' not in failed.text
    recovered = client.post('/api/v1/query', headers=headers, json={'question': 'Next'})
    assert recovered.status_code == 200 and recovered.json()['answer'] == 'Recovered'


def test_demo_token_expires_at_exact_boundary_and_revocation_survives_restart(tmp_path, monkeypatch):
    class Clock(datetime):
        current = datetime(2026, 1, 1, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            return cls.current if tz is not None else cls.current.replace(tzinfo=None)

    monkeypatch.setattr(api_tokens, 'datetime', Clock)
    path = tmp_path / 'tokens.sqlite3'
    registry = api_tokens.TokenRegistry(path)
    expiring = registry.create('expires', 'demo')
    revoked = registry.create('revoked', 'demo')
    owner = registry.create('owner')
    assert registry.revoke(revoked['id'])

    Clock.current += timedelta(days=7, microseconds=-1)
    reopened = api_tokens.TokenRegistry(path)
    assert reopened.scope(expiring['token']) == 'demo'
    assert reopened.scope(revoked['token']) is None
    Clock.current += timedelta(microseconds=1)
    assert reopened.scope(expiring['token']) is None
    assert reopened.scope(owner['token']) == 'full'
    assert all('token' not in item and 'digest' not in item for item in reopened.list())
