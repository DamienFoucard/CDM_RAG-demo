"""First-party MCP tools over the existing CDM RAG query and status API."""
from __future__ import annotations

import asyncio
import ipaddress
import os
from typing import Any, Annotated, Protocol
from urllib.parse import urlsplit

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field

TIMEOUT_SECONDS = 600
MAX_ANSWER = 12000
MAX_SOURCES = 12
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class Backend(Protocol):
    async def query(self, question: str) -> dict[str, Any]: ...
    async def status(self) -> dict[str, Any]: ...


def _string(value: Any, limit: int) -> str:
    return value[:limit] if isinstance(value, str) else ''


def _project_status(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {'status': 'catalog_ready'}
    for key in ('entity_count', 'passage_count'):
        value = data.get(key)
        result[key] = value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 100000000 else None
    for key in ('source_commit', 'index_revision'):
        result[key] = _string(data.get(key), 128)
    return result


def _project_answer(data: dict[str, Any]) -> dict[str, Any]:
    raw_answer = data.get('answer')
    if not isinstance(raw_answer, str):
        raise ServiceError('The CDM RAG service returned an invalid answer.')
    raw_sources = data.get('sources') or []
    if not isinstance(raw_sources, list) or any(not isinstance(source, dict) for source in raw_sources):
        raise ServiceError('The CDM RAG service returned invalid source evidence.')

    # Every source in the query result backs a displayed citation. Never return a
    # partial answer whose citations cannot all be matched to their evidence.
    limits = {'citation': 128, 'display_citation': 2048, 'title': 512, 'source_id': 1024,
              'source_type': 128, 'source_label': 1024, 'entity_id': 1024, 'kind': 128,
              'passage_id': 1024, 'path': 2048, 'uri': 2048}
    exceeds_limit = (len(raw_answer) > MAX_ANSWER or len(raw_sources) > MAX_SOURCES
                     or any(isinstance(source.get(key), str) and len(source[key]) > limit
                            for source in raw_sources for key, limit in limits.items()))
    if exceeds_limit:
        return {'status': 'result_limit_exceeded', 'answer': '', 'sources': [],
                'source_count': len(raw_sources), 'sources_truncated': bool(raw_sources),
                'answer_truncated': bool(raw_answer),
                'message': 'The complete answer and evidence exceed MCP response limits; use the HTTP API.'}

    sources = []
    for source in raw_sources:
        projected = {key: source[key] for key in limits if isinstance(source.get(key), str)}
        if isinstance(source.get('text'), str):
            projected['text'] = source['text'][:800]
            projected['text_truncated'] = len(source['text']) > 800
        sources.append(projected)
    route = _string(data.get('route'), 80)
    result: dict[str, Any] = {'status': 'ok' if sources else 'insufficient_evidence',
                              'answer': raw_answer, 'sources': sources, 'route': route,
                              'source_count': len(sources), 'sources_truncated': False,
                              'answer_truncated': False,
                              'query_id': _string(data.get('query_id'), 128)}
    processing = data.get('processing')
    if isinstance(processing, dict):
        # Avoid returning the complete configuration or private runtime settings.
        result['processing'] = {key: processing[key] for key in ('elapsed_ms', 'queue_ms')
                                if isinstance(processing.get(key), (int, float)) and not isinstance(processing[key], bool)}
        if isinstance(processing.get('answer_method'), str):
            result['processing']['answer_method'] = _string(processing['answer_method'], 160)
        calls = processing.get('model_calls')
        if isinstance(calls, list):
            result['processing']['model_calls'] = [
                {k: _string(call.get(k), 160) for k in ('role', 'model', 'stage', 'outcome')
                 if isinstance(call.get(k), str)} for call in calls[:32] if isinstance(call, dict)]
    return result


class ServiceError(Exception):
    pass


class HTTPBackend:
    """Call a configured API origin; never forward credentials to redirects."""
    def __init__(self, base_url: str, token: str = ''):
        try:
            parts = urlsplit(base_url)
            port = parts.port
        except ValueError as exc:
            raise ValueError('Invalid API origin.') from exc
        host = parts.hostname or ''
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.lower() == 'localhost'
        if (parts.scheme not in {'http', 'https'} or not host or parts.username is not None
                or parts.password is not None or parts.query or parts.fragment or parts.path not in {'', '/'}
                or (parts.scheme == 'http' and not loopback) or (not loopback and not token)
                or (port is not None and not 1 <= port <= 65535)):
            raise ValueError('Use a loopback HTTP origin, or an HTTPS origin and API token.')
        if token and (len(token) > 512 or any(ord(c) < 33 or ord(c) > 126 for c in token)):
            raise ValueError('Invalid API token.')
        self.base_url = base_url.rstrip('/')
        self._token = token

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        headers = {'Accept': 'application/json', 'User-Agent': 'CDM-RAG-MCP/1.0'}
        if self._token:
            headers['Authorization'] = 'Bearer ' + self._token
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=False, trust_env=False) as client:
                async with client.stream(method, self.base_url + path, headers=headers, **kwargs) as response:
                    if response.status_code in (401, 403):
                        raise ServiceError('Authentication failed. Check the API token and access.')
                    if response.status_code != 200:
                        raise ServiceError('The CDM RAG service could not complete the request.')
                    body = bytearray()
                    async for part in response.aiter_bytes():
                        body.extend(part)
                        if len(body) > MAX_RESPONSE_BYTES:
                            raise ServiceError('The CDM RAG response exceeded the size limit.')
            import json
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ServiceError('The CDM RAG service returned an invalid response.')
            return data
        except httpx.TimeoutException as exc:
            raise ServiceError('The CDM RAG service timed out.') from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ServiceError('The CDM RAG service is unavailable or returned an invalid response.') from exc

    async def query(self, question: str) -> dict[str, Any]:
        return await self._request('POST', '/api/v1/query', json={'question': question})

    async def status(self) -> dict[str, Any]:
        return await self._request('GET', '/api/v1/status')


class AppBackend:
    def __init__(self, app: Any):
        self.app = app

    async def query(self, question: str) -> dict[str, Any]:
        # Import late to avoid a cycle with the host application's lifespan.
        from .main import Question, _run_logged_query
        from .api_tokens import run_query_worker
        from .observability import new_query_id
        try:
            return await asyncio.wait_for(run_query_worker(_run_logged_query, Question(question=question), new_query_id()),
                                          timeout=TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            raise ServiceError('The CDM RAG service timed out.') from exc
        except Exception as exc:
            raise ServiceError('The CDM RAG service could not complete the request.') from exc

    async def status(self) -> dict[str, Any]:
        try:
            index = self.app.state.index
            return {'entity_count': len(index.catalog['entities']), 'passage_count': len(index.chunks),
                    'source_commit': index.catalog['source_commit'], 'index_revision': index.index_revision}
        except (AttributeError, KeyError, TypeError) as exc:
            raise ServiceError('The CDM RAG service is not ready.') from exc


def create_server(backend: Backend, *, public_host: str = '') -> FastMCP:
    hosts = ['localhost:*', '127.0.0.1:*', '[::1]:*', 'localhost', '127.0.0.1', '[::1]']
    origins = ['http://localhost:*', 'http://127.0.0.1:*', 'http://[::1]:*']
    if public_host:
        parsed = urlsplit('https://' + public_host)
        if parsed.hostname != public_host or parsed.port is not None:
            raise ValueError('CDM_RAG_PUBLIC_HOST must be a hostname without a port.')
        hosts.append(public_host)
        origins.append('https://' + public_host)
    mcp = FastMCP('CDM RAG', streamable_http_path='/', stateless_http=True, json_response=True,
                  max_request_body_size=131072,
                  transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                                               allowed_hosts=hosts, allowed_origins=origins))
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    @mcp.tool(annotations=annotations)
    async def ask_schema_question(question: Annotated[str, Field(min_length=1, max_length=2000)]) -> dict[str, Any]:
        """Ask about CDM schema definitions and return the answer and source pointers."""
        question = ' '.join(question.split())
        if not question:
            raise ToolError('Question must contain non-whitespace text.')
        try:
            return _project_answer(await backend.query(question))
        except ServiceError as exc:
            raise ToolError(str(exc)) from None
        except Exception:
            raise ToolError('The CDM RAG service could not complete the request.') from None

    @mcp.tool(annotations=annotations)
    async def service_status() -> dict[str, Any]:
        """Report catalog readiness and source revision without private runtime settings."""
        try:
            return _project_status(await backend.status())
        except ServiceError as exc:
            raise ToolError(str(exc)) from None
        except Exception:
            raise ToolError('The CDM RAG service is unavailable.') from None

    return mcp
