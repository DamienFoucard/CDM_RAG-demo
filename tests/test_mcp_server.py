"""Protocol and security checks for the first-party MCP service."""
import asyncio
import json
import threading
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from app.api_tokens import APIAuthMiddleware, TokenRegistry
from app.mcp_server import HTTPBackend, ServiceError, create_server, _project_answer
from app.rag import format_entity_citations

TOKEN = 't' * 40
SECRET = 'a' * 64


class Backend:
    async def status(self):
        return {'entity_count': 2, 'passage_count': 4, 'source_commit': 'abc', 'index_revision': 'def',
                'query_log_path': '/private/log', 'answer_model': 'private'}

    async def query(self, question):
        return {'answer': 'A sourced answer', 'route': 'cdm', 'sources': [
            {'citation': 'R1', 'uri': 'https://example.org/definition', 'text': 'evidence',
             'passage_id': 'P1', 'token': 'never-return'}],
            'processing': {'elapsed_ms': 23, 'configured_models': {'private': 'setting'},
                           'model_calls': [{'role': 'scope', 'model': 'model:1', 'stage': 'checking_scope',
                                            'outcome': 'completed'}]}}


def test_in_memory_mcp_initialize_discover_and_call():
    async def run():
        async with create_connected_server_and_client_session(create_server(Backend())) as session:
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {'ask_schema_question', 'service_status'}
            status = await session.call_tool('service_status', {})
            assert not status.isError
            assert status.structuredContent['entity_count'] == 2
            assert 'query_log_path' not in str(status)
            result = await session.call_tool('ask_schema_question', {'question': 'What is Account?'})
            assert not result.isError
            assert result.structuredContent['sources'][0]['passage_id'] == 'P1'
            assert 'never-return' not in str(result)
            assert 'configured_models' not in str(result)
            bad = await session.call_tool('ask_schema_question', {'question': ' '})
            assert bad.isError
            invalid = await session.call_tool('ask_schema_question', {'question': 'x' * 2001})
            assert invalid.isError
    asyncio.run(run())


def test_errors_are_clear_and_do_not_leak_internal_details():
    class Failed(Backend):
        async def query(self, question):
            raise ServiceError('The CDM RAG service timed out.')
    async def run():
        async with create_connected_server_and_client_session(create_server(Failed())) as session:
            result = await session.call_tool('ask_schema_question', {'question': 'Account?'})
            assert result.isError and 'timed out' in str(result)
    asyncio.run(run())


def test_projected_evidence_preserves_real_display_citations_for_same_entity():
    source_data = [
        {'citation': 'R1', 'entity_id': 'Model::Account', 'source_type': 'entity', 'kind': 'summary',
         'passage_id': 'Model::Account:1', 'uri': 'https://example.org/account', 'text': 'A' * 801},
        {'citation': 'R2', 'entity_id': 'Model::Account', 'source_type': 'entity', 'kind': 'attributes',
         'passage_id': 'Model::Account:2', 'uri': 'https://example.org/account', 'text': 'more evidence'},
    ]
    answer, cited_sources = format_entity_citations('First fact [R1]. Second fact [R2].', source_data)
    result = _project_answer({'answer': answer, 'sources': cited_sources, 'route': 'cdm'})
    assert result['status'] == 'ok' and not result['answer_truncated'] and not result['sources_truncated']
    assert [source['display_citation'] for source in result['sources']] == [
        '[Model::Account · R1]', '[Model::Account · R2]']
    assert all(source['display_citation'] in result['answer'] for source in result['sources'])
    assert result['sources'][0]['entity_id'] == 'Model::Account'
    assert result['sources'][1]['kind'] == 'attributes'
    assert result['sources'][0]['text_truncated'] and len(result['sources'][0]['text']) == 800


def test_oversized_answer_or_evidence_is_withheld_instead_of_breaking_citations():
    many = [{'citation': f'R{i}', 'display_citation': f'[Model::A · R{i}]',
             'entity_id': 'Model::A', 'uri': 'https://example.org/a'} for i in range(13)]
    result = _project_answer({'answer': ' '.join(s['display_citation'] for s in many), 'sources': many})
    assert result['status'] == 'result_limit_exceeded'
    assert result['answer'] == '' and result['sources'] == []
    assert result['sources_truncated'] and result['source_count'] == 13
    result = _project_answer({'answer': 'x' * 12001 + '[Model::A]', 'sources': many[:1]})
    assert result['status'] == 'result_limit_exceeded' and result['answer_truncated']
    assert result['answer'] == '' and result['sources'] == []


def test_hosted_transport_requires_bearer_and_validates_host_and_origin(tmp_path):
    server = create_server(Backend(), public_host='cdm-rag.example.com')
    mcp_app = server.streamable_http_app()
    @asynccontextmanager
    async def lifespan(app):
        async with server.session_manager.run():
            yield
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(APIAuthMiddleware, required=True, registry=TokenRegistry(tmp_path / 'tokens.db', TOKEN),
                       proxy_secret=SECRET, owner_email='owner@example.com', owner_subject='owner-sub')
    app.mount('/api/v1/mcp', mcp_app)
    payload = {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {
        'protocolVersion': '2025-06-18', 'capabilities': {},
        'clientInfo': {'name': 'test-client', 'version': '1'}}}
    headers = {'Host': 'cdm-rag.example.com', 'Accept': 'application/json, text/event-stream'}
    with TestClient(app, base_url='https://cdm-rag.example.com') as client:
        url = '/api/v1/mcp/'
        assert client.post(url, json=payload, headers=headers).status_code == 401
        assert client.post(url, json=payload, headers={**headers, 'X-CDM-RAG-Owner-Session': SECRET,
                                                       'X-CDM-RAG-Owner-Email': 'owner@example.com'}).status_code == 401
        assert client.post(url, json=payload, headers={**headers, 'Authorization': 'Bearer invalid'}).status_code == 401
        good = {**headers, 'Authorization': 'Bearer ' + TOKEN}
        assert client.post(url, json=payload, headers={**good, 'Host': 'attacker.example'}).status_code in (400, 403, 421)
        assert client.post(url, json=payload, headers={**good, 'Origin': 'https://attacker.example'}).status_code in (400, 403, 421)
        init = client.post(url, json=payload, headers=good)
        assert init.status_code == 200, init.text
        assert init.json()['result']['protocolVersion']
        listed = client.post(url, json={'jsonrpc':'2.0','id':2,'method':'tools/list','params':{}}, headers=good)
        assert listed.status_code == 200, listed.text
        assert {t['name'] for t in listed.json()['result']['tools']} == {'ask_schema_question', 'service_status'}
        called = client.post(url, json={'jsonrpc':'2.0','id':3,'method':'tools/call',
                                       'params':{'name':'service_status','arguments':{}}}, headers=good)
        assert called.status_code == 200 and called.json()['result']['structuredContent']['entity_count'] == 2


def test_demo_mcp_tool_inherits_request_lease(tmp_path):
    from datetime import datetime, timedelta, timezone
    from app.api_tokens import _demo_lease, run_query_worker
    seen = []

    class DemoBackend(Backend):
        async def query(self, question):
            seen.append(_demo_lease.get() is not None)
            await run_query_worker(lambda: None)
            return {'answer': 'answer', 'sources': []}

    server = create_server(DemoBackend(), public_host='cdm-rag.example.com')
    @asynccontextmanager
    async def lifespan(app):
        async with server.session_manager.run():
            yield
    registry = TokenRegistry(tmp_path / 'demo.db')
    token = registry.create('demo', 'demo')['token']
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(APIAuthMiddleware, required=True, registry=registry,
                       proxy_secret=SECRET, owner_email='owner@example.com', owner_subject='owner-sub',
                       demo_expires_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat())
    app.mount('/api/v1/mcp', server.streamable_http_app())
    headers = {'Host': 'cdm-rag.example.com', 'Accept': 'application/json, text/event-stream',
               'Authorization': 'Bearer ' + token}
    payload = {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call',
               'params': {'name': 'ask_schema_question', 'arguments': {'question': 'What is Account?'}}}
    with TestClient(app, base_url='https://cdm-rag.example.com') as client:
        called = client.post('/api/v1/mcp/', json=payload, headers=headers)
        assert called.status_code == 200, called.text
        assert seen == [True]


def test_http_backend_requires_secure_origin_and_token():
    HTTPBackend('http://127.0.0.1:8124')
    HTTPBackend('https://cdm-rag.example.com', TOKEN)
    for url, token in [('http://example.com',''), ('https://example.com',''),
                       ('https://user:secret@example.com', TOKEN), ('http://127.0.0.1:8124/path', '')]:
        with pytest.raises(ValueError):
            HTTPBackend(url, token)


def test_stdio_protocol_with_local_api():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            data = json.dumps({'entity_count': 2, 'passage_count': 3, 'source_commit':'abc',
                               'index_revision':'xyz', 'private_setting':'never-return'}).encode()
            self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(data)
        def log_message(self, *args):
            pass
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
    async def run():
        import os, sys
        params = StdioServerParameters(command=sys.executable, args=['-m', 'app.mcp_stdio'],
                                       env={**os.environ, 'CDM_RAG_MCP_BASE_URL': f'http://127.0.0.1:{http.server_port}'})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                assert len((await session.list_tools()).tools) == 2
                result = await session.call_tool('service_status', {})
                assert result.structuredContent['entity_count'] == 2
                assert 'never-return' not in str(result)
    try:
        asyncio.run(run())
    finally:
        http.shutdown(); http.server_close(); thread.join()


def test_remote_backend_auth_failures_redirects_and_errors_are_bounded(monkeypatch):
    client_cls = httpx.AsyncClient
    seen = []
    def make_client(**kwargs):
        def handle(request):
            seen.append(request)
            return httpx.Response(401, text='secret internal error')
        return client_cls(transport=httpx.MockTransport(handle), **kwargs)
    monkeypatch.setattr('app.mcp_server.httpx.AsyncClient', make_client)
    async def run():
        backend = HTTPBackend('https://cdm-rag.example.com', TOKEN)
        with pytest.raises(ServiceError, match='Authentication failed') as caught:
            await backend.status()
        assert TOKEN not in str(caught.value)
        assert seen[0].headers['Authorization'] == 'Bearer ' + TOKEN
    asyncio.run(run())


def test_insufficient_evidence_and_unexpected_errors_are_safe():
    class NoSources(Backend):
        async def query(self, question):
            return {'answer': 'No evidence', 'sources': [], 'route': 'no_search'}
    class Unexpected(Backend):
        async def status(self):
            raise RuntimeError('private token or internal detail')
    async def run():
        async with create_connected_server_and_client_session(create_server(NoSources())) as session:
            result = await session.call_tool('ask_schema_question', {'question':'unrelated?'})
            assert result.structuredContent['status'] == 'insufficient_evidence'
        async with create_connected_server_and_client_session(create_server(Unexpected())) as session:
            result = await session.call_tool('service_status', {})
            assert result.isError and 'private token or internal detail' not in str(result)
    asyncio.run(run())


def test_official_mcp_client_over_streamable_http(tmp_path):
    import socket, time
    import uvicorn
    from mcp.client.streamable_http import streamable_http_client
    server = create_server(Backend())
    mcp_app = server.streamable_http_app()
    @asynccontextmanager
    async def lifespan(app):
        async with server.session_manager.run():
            yield
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(APIAuthMiddleware, required=True, registry=TokenRegistry(tmp_path / 'tokens.db', TOKEN),
                       proxy_secret=SECRET, owner_email='owner@example.com', owner_subject='owner-sub')
    app.mount('/api/v1/mcp', mcp_app)
    sock = socket.socket(); sock.bind(('127.0.0.1', 0)); sock.listen(128)
    port = sock.getsockname()[1]
    uv = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    thread = threading.Thread(target=lambda: uv.run(sockets=[sock]), daemon=True); thread.start()
    try:
        deadline = time.monotonic() + 10
        while not uv.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert uv.started
        async def run():
            async with httpx.AsyncClient(headers={'Authorization':'Bearer '+TOKEN}) as http_client:
                async with streamable_http_client(f'http://127.0.0.1:{port}/api/v1/mcp/',
                                                  http_client=http_client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        assert {t.name for t in (await session.list_tools()).tools} == {'ask_schema_question', 'service_status'}
                        result = await session.call_tool('service_status', {})
                        assert not result.isError and result.structuredContent['entity_count'] == 2
        asyncio.run(run())
    finally:
        uv.should_exit = True
        thread.join(timeout=10)
        sock.close()
