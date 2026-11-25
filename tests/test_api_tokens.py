from fastapi import FastAPI
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
import pytest
import sqlite3
import threading
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from app.api_tokens import APIAuthMiddleware, TokenRegistry, run_query_worker

DEMO_DEADLINE = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()


def test_registry_stores_only_hashes_and_revocation_survives_restart(tmp_path):
    initial = 'initial-' + 'x' * 48
    path = tmp_path / 'registry.sqlite3'
    registry = TokenRegistry(path, initial)
    created = registry.create('integration application')
    assert registry.valid(initial)
    assert registry.valid(created['token'])
    assert initial.encode() not in path.read_bytes()
    assert created['token'].encode() not in path.read_bytes()
    assert all('token' not in entry and 'digest' not in entry for entry in registry.list())
    initial_id = next(entry['id'] for entry in registry.list() if entry['label'] == 'owner')
    assert registry.revoke(initial_id)
    assert registry.revoke(created['id'])
    assert not registry.valid(initial)
    assert not registry.valid(created['token'])
    assert not registry.revoke(created['id'])
    assert not TokenRegistry(path, initial).valid(initial)


def test_api_authentication_and_session_only_management(tmp_path):
    registry = TokenRegistry(tmp_path / 'tokens.sqlite3')
    created = registry.create('app')
    secret = 'a' * 64
    app = FastAPI()
    app.add_middleware(APIAuthMiddleware, required=True, registry=registry, proxy_secret=secret, owner_email='owner@example.test', owner_subject='owner-sub', demo_expires_at=DEMO_DEADLINE)

    @app.get('/api/v1/status')
    def status(): return {'ok': True}

    @app.get('/api/v1/api-tokens')
    def tokens(): return {'ok': True}

    @app.get('/public')
    def public(): return {'ok': True}

    client = TestClient(app)
    assert client.get('/public').status_code == 200
    assert client.get('/api/v1/status').status_code == 401
    assert client.get('/api/v1/status', headers={'X-CDM-RAG-Owner-Session': 'fake'}).status_code == 401
    assert client.get('/api/v1/status', headers={'Authorization': 'Bearer wrong'}).status_code == 401
    assert client.get('/api/v1/status', headers={'Authorization': 'Bearer ' + created['token']}).status_code == 200
    assert client.get('/api/v1/api-tokens', headers={'Authorization': 'Bearer ' + created['token']}).status_code == 401
    assert client.get('/api/v1/api-tokens', headers={'X-CDM-RAG-Owner-Session': secret}).status_code == 401
    assert client.get('/api/v1/api-tokens', headers={'X-CDM-RAG-Owner-Session': secret, 'X-CDM-RAG-Owner-Email': 'someone@example.test'}).status_code == 401
    assert client.get('/api/v1/api-tokens', headers={'X-CDM-RAG-Owner-Session': secret, 'X-CDM-RAG-Owner-Email': 'owner@example.test'}).status_code == 401
    assert client.get('/api/v1/api-tokens', headers={'X-CDM-RAG-Owner-Session': secret, 'X-CDM-RAG-Owner-Email': 'owner@example.test', 'X-CDM-RAG-Owner-Subject': 'owner-sub'}).status_code == 200
    assert registry.revoke(created['id'])
    assert client.get('/api/v1/status', headers={'Authorization': 'Bearer ' + created['token']}).status_code == 401


def test_management_origin(monkeypatch, tmp_path):
    import app.main as main
    monkeypatch.setenv('CDM_RAG_PUBLIC_HOST', 'example.test')
    monkeypatch.setattr(main, 'API_TOKEN_REGISTRY', TokenRegistry(tmp_path / 'tokens.sqlite3'))
    client = TestClient(main.app)
    assert client.post('/api/v1/api-tokens', json={'label': 'x'}).status_code == 403
    headers = {'Origin': 'https://example.test'}
    result = client.post('/api/v1/api-tokens', json={'label': '  App A  '}, headers=headers)
    assert result.status_code == 201
    assert result.headers['cache-control'] == 'no-store'
    token = result.json()
    assert token['label'] == 'App A'
    stored = client.get('/api/v1/api-tokens').json()['tokens']
    assert token['token'] not in str(stored)
    assert client.delete('/api/v1/api-tokens/' + token['id'], headers=headers).status_code == 204
    assert not main.API_TOKEN_REGISTRY.valid(token['token'])


def test_legacy_table_remains_rollback_compatible_and_demo_tokens_stay_separate(tmp_path):
    path = tmp_path / 'legacy.sqlite3'
    import hashlib
    old = 'old-' + 'x' * 48
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE api_tokens (id TEXT PRIMARY KEY, label TEXT NOT NULL, digest TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL, revoked_at TEXT)')
        db.execute('INSERT INTO api_tokens VALUES (?, ?, ?, ?, NULL)', ('old-id', 'old', hashlib.sha256(old.encode()).hexdigest(), 'now'))
    registry = TokenRegistry(path)
    demo = registry.create('demo', 'demo')
    with sqlite3.connect(path) as db:
        assert len(db.execute('PRAGMA table_info(api_tokens)').fetchall()) == 5
        # The previously deployed release uses a five-value seed INSERT.
        db.execute('INSERT OR IGNORE INTO api_tokens VALUES (?, ?, ?, ?, NULL)',
                   ('second-id', 'old', hashlib.sha256(old.encode()).hexdigest(), 'now'))
        assert db.execute('SELECT 1 FROM api_tokens WHERE digest=?',
                          (hashlib.sha256(demo['token'].encode()).hexdigest(),)).fetchone() is None
    assert registry.scope(old) == 'full'
    assert registry.scope(demo['token']) == 'demo'
    with pytest.raises(ValueError):
        TokenRegistry(path, demo['token'])
    assert next(x for x in registry.list() if x['id'] == demo['id'])['scope'] == 'demo'
    assert registry.revoke(demo['id'])
    assert registry.scope(demo['token']) is None


def test_demo_session_and_token_are_allowlisted_and_owner_keeps_access(tmp_path):
    registry = TokenRegistry(tmp_path / 'tokens.sqlite3')
    full, demo = registry.create('full'), registry.create('demo', 'demo')
    secret = 'a' * 64
    app = FastAPI()
    app.add_middleware(APIAuthMiddleware, required=True, registry=registry, proxy_secret=secret, owner_email='owner@example.test', owner_subject='owner-sub', demo_expires_at=DEMO_DEADLINE)

    @app.api_route('/{path:path}', methods=['GET', 'POST', 'PUT', 'DELETE'])
    def any_route(path: str): return {'ok': True}

    client = TestClient(app)
    owner = {'X-CDM-RAG-Owner-Session': secret, 'X-CDM-RAG-Owner-Email': 'owner@example.test', 'X-CDM-RAG-Owner-Subject': 'owner-sub'}
    guest = {'X-CDM-RAG-Owner-Session': secret, 'X-CDM-RAG-Owner-Email': 'demo@example.test', 'X-CDM-RAG-Owner-Subject': 'demo-sub'}
    impersonating_email = {**guest, 'X-CDM-RAG-Owner-Email': 'owner@example.test'}
    assert client.put('/api/v1/settings/models', headers=impersonating_email).status_code == 403
    assert client.get('/api/v1/api-tokens', headers=impersonating_email).status_code == 401
    bearer = {'Authorization': 'Bearer ' + demo['token']}
    full_bearer = {'Authorization': 'Bearer ' + full['token']}
    for headers in (guest, bearer):
        assert client.get('/api/v1/status', headers=headers).status_code == 200
        assert client.post('/api/v1/query', headers=headers).status_code == 200
        assert client.post('/api/v1/query/stream', headers=headers).status_code == 200
        assert client.get('/api/v1/graph/search', headers=headers).status_code == 200
        assert client.put('/api/v1/settings/models', headers=headers).status_code == 403
        assert client.get('/api/v1/benchmark/runs', headers=headers).status_code == 403
        assert client.post('/api/v1/benchmark/stream', headers=headers).status_code == 403
        assert client.get('/api/v1/retrieval-benchmark/runs', headers=headers).status_code == 403
        assert client.get('/api/v1/api-tokens', headers=headers).status_code == 401
        assert client.get('/api/v1/new-future-route', headers=headers).status_code == 403
    assert client.get('/home', headers=guest).status_code == 200
    assert client.get('/RAG', headers=guest).status_code == 200
    assert client.get('/benchmarks', headers=guest).status_code == 403
    assert client.get('/evaluation/results/sensitive.json', headers=guest).status_code == 403
    assert client.get('/api-tokens', headers=guest).status_code == 403
    assert client.post('/api/v1/mcp/', headers=guest).status_code == 401
    assert client.post('/api/v1/mcp/', headers=bearer).status_code == 200
    assert client.put('/api/v1/settings/models', headers=owner).status_code == 200
    assert client.get('/api/v1/api-tokens', headers=owner).status_code == 200
    assert client.put('/api/v1/settings/models', headers=full_bearer).status_code == 200
    assert client.get('/api/v1/status', headers={'X-CDM-RAG-Owner-Session': secret}).status_code == 401
    for path in ('/api/v1/%73ettings/models', '/api//v1/settings/models',
                 '/api/v1/query%2f..%2fsettings/models', '/api/%2e%2e/evaluation/results/x'):
        assert client.get(path, headers=bearer).status_code in {401, 403, 404}


def test_demo_query_limit_is_shared_and_does_not_restrict_owner(tmp_path):
    registry = TokenRegistry(tmp_path / 'tokens.sqlite3')
    first, second = registry.create('first', 'demo'), registry.create('second', 'demo')
    app = FastAPI()
    app.add_middleware(APIAuthMiddleware, required=True, registry=registry, proxy_secret='a' * 64, owner_email='owner@example.test', owner_subject='owner-sub', demo_expires_at=DEMO_DEADLINE)

    @app.post('/api/v1/query')
    def query(): return {'ok': True}

    client = TestClient(app)
    for _ in range(12):
        assert client.post('/api/v1/query', headers={'Authorization': 'Bearer ' + first['token']}).status_code == 200
    limited = client.post('/api/v1/query', headers={'Authorization': 'Bearer ' + second['token']})
    assert limited.status_code == 429
    assert int(limited.headers['retry-after']) > 0
    assert client.post('/api/v1/query', headers={'X-CDM-RAG-Owner-Session': 'a' * 64,
                                                'X-CDM-RAG-Owner-Email': 'owner@example.test', 'X-CDM-RAG-Owner-Subject': 'owner-sub'}).status_code == 200


def test_demo_token_can_be_created_only_with_valid_scope(monkeypatch, tmp_path):
    import app.main as main
    monkeypatch.setenv('CDM_RAG_PUBLIC_HOST', 'example.test')
    monkeypatch.setattr(main, 'API_TOKEN_REGISTRY', TokenRegistry(tmp_path / 'tokens.sqlite3'))
    client = TestClient(main.app)
    headers = {'Origin': 'https://example.test'}
    created = client.post('/api/v1/api-tokens', json={'label': 'demo', 'scope': 'demo'}, headers=headers)
    assert created.status_code == 201
    assert created.json()['scope'] == 'demo'
    assert main.API_TOKEN_REGISTRY.scope(created.json()['token']) == 'demo'
    assert client.post('/api/v1/api-tokens', json={'label': 'bad', 'scope': 'unrestricted'}, headers=headers).status_code == 422


def test_demo_rejects_concurrent_queries_and_releases_slot(tmp_path):
    registry = TokenRegistry(tmp_path / 'tokens.sqlite3')
    demo = registry.create('demo', 'demo')
    app = FastAPI()
    app.add_middleware(APIAuthMiddleware, required=True, registry=registry, proxy_secret='a' * 64, owner_email='owner@example.test', owner_subject='owner-sub', demo_expires_at=DEMO_DEADLINE)
    started, release = threading.Event(), threading.Event()

    @app.post('/api/v1/query')
    def query():
        started.set()
        assert release.wait(5)
        return {'ok': True}

    client = TestClient(app)
    headers = {'Authorization': 'Bearer ' + demo['token']}
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, '/api/v1/query', headers=headers)
        assert started.wait(5)
        try:
            assert client.post('/api/v1/query', headers=headers).status_code == 429
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 200
    assert client.post('/api/v1/query', headers=headers).status_code == 200


def test_detached_query_and_mcp_workers_keep_slot_until_thread_finishes(tmp_path):
    registry = TokenRegistry(tmp_path / 'tokens.sqlite3')
    demo = registry.create('demo', 'demo')
    app = FastAPI()
    app.add_middleware(APIAuthMiddleware, required=True, registry=registry, proxy_secret='a' * 64, owner_email='owner@example.test', owner_subject='owner-sub', demo_expires_at=DEMO_DEADLINE)
    started, release = threading.Event(), threading.Event()

    def worker():
        started.set()
        assert release.wait(5)

    @app.post('/api/v1/query/stream')
    @app.post('/api/v1/mcp/')
    async def detached():
        task = asyncio.create_task(run_query_worker(worker))
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()  # Simulates a disconnect or MCP timeout after submission.
        try:
            await task
        except asyncio.CancelledError:
            pass
        return {'ok': True}

    client = TestClient(app)
    headers = {'Authorization': 'Bearer ' + demo['token']}
    for path in ('/api/v1/query/stream', '/api/v1/mcp/'):
        started.clear(); release.clear()
        assert client.post(path, headers=headers).status_code == 200
        try:
            assert client.post(path, headers=headers).status_code == 429
        finally:
            release.set()
        # The concurrent-futures done callback runs on the worker thread.
        deadline = time.monotonic() + 3
        response = None
        while time.monotonic() < deadline:
            response = client.post(path, headers=headers)
            if response.status_code != 429:
                break
            time.sleep(0.01)
        assert response is not None and response.status_code == 200


def test_demo_expiration_is_fail_closed_without_affecting_full_tokens(tmp_path):
    registry = TokenRegistry(tmp_path / 'tokens.sqlite3')
    demo, full = registry.create('demo', 'demo'), registry.create('full')
    assert demo['expires_at'] and registry.scope(demo['token']) == 'demo'
    for expiry in ('invalid', '2020-01-01T00:00:00+00:00', None):
        with sqlite3.connect(registry.path) as db:
            db.execute('UPDATE demo_api_tokens SET expires_at=? WHERE id=?', (expiry, demo['id']))
        assert registry.scope(demo['token']) is None
    assert registry.scope(full['token']) == 'full'
    demo2 = registry.create('demo2', 'demo')
    for deadline in ('', 'invalid', '2020-01-01T00:00:00Z'):
        app = FastAPI()
        app.add_middleware(APIAuthMiddleware, required=True, registry=registry, proxy_secret='a' * 64, owner_email='owner@example.test', owner_subject='owner-sub', demo_expires_at=deadline)
        @app.get('/api/v1/status')
        def status(): return {'ok': True}
        client = TestClient(app)
        assert client.get('/api/v1/status', headers={'Authorization': 'Bearer ' + demo2['token']}).status_code == 403
        assert client.get('/api/v1/status', headers={'Authorization': 'Bearer ' + full['token']}).status_code == 200


@pytest.mark.parametrize('rerank_default', [False, True])
def test_demo_query_options_follow_server_reranking_default(monkeypatch, rerank_default):
    from app import main
    from app.main import Question, check_demo_query_options
    monkeypatch.setattr(main, 'DEFAULT_RERANK_ENABLED', rerank_default)
    demo = Request({'type': 'http', 'state': {'demo': True}})
    owner = Request({'type': 'http', 'state': {'demo': False}})
    check_demo_query_options(demo, Question(question='q', rerank=rerank_default))
    check_demo_query_options(demo, Question(question='q', rerank=False))
    if not rerank_default:
        with pytest.raises(HTTPException) as exc:
            check_demo_query_options(demo, Question(question='q', rerank=True))
        assert exc.value.status_code == 403
    for options in ({'limit': 12}, {'mode': 'semantic'}):
        body = Question(question='q', **options)
        with pytest.raises(HTTPException) as exc:
            check_demo_query_options(demo, body)
        assert exc.value.status_code == 403
        check_demo_query_options(owner, body)
