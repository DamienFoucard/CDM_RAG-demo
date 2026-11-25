"""Private API token registry and ASGI authentication middleware."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import asyncio
import contextvars
import functools
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deadline(value: str | None) -> datetime | None:
    try:
        parsed = datetime.fromisoformat((value or '').replace('Z', '+00:00'))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


class TokenRegistry:
    def __init__(self, path: Path, initial_token: str | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS api_tokens (
                id TEXT PRIMARY KEY, label TEXT NOT NULL, digest TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL, revoked_at TEXT)''')
            # Leave the deployed five-column table unchanged. Older releases
            # cannot authenticate scoped demo tokens even after rollback.
            db.execute('''CREATE TABLE IF NOT EXISTS demo_api_tokens (
                id TEXT PRIMARY KEY, label TEXT NOT NULL, digest TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL, revoked_at TEXT, expires_at TEXT)''')
            if initial_token:
                if len(initial_token) < 32:
                    raise ValueError('Initial API token must be at least 32 characters.')
                if db.execute('SELECT 1 FROM demo_api_tokens WHERE digest = ?', (_digest(initial_token),)).fetchone():
                    raise ValueError('Initial API token must not be a demo token.')
                db.execute('INSERT OR IGNORE INTO api_tokens (id, label, digest, created_at, revoked_at) VALUES (?, ?, ?, ?, NULL)',
                           (str(uuid4()), 'owner', _digest(initial_token), _now()))
        os.chmod(path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def create(self, label: str, scope: str = 'full') -> dict:
        if scope not in {'full', 'demo'}:
            raise ValueError('Invalid token scope')
        token = ('cdmd_' if scope == 'demo' else 'cdmr_') + secrets.token_urlsafe(32)
        identifier, created = str(uuid4()), _now()
        expires_at = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat() if scope == 'demo' else None
        with self._connect() as db:
            if scope == 'demo':
                if db.execute('SELECT 1 FROM api_tokens WHERE digest = ?', (_digest(token),)).fetchone():
                    raise ValueError('Token already exists')
                db.execute('INSERT INTO demo_api_tokens VALUES (?, ?, ?, ?, NULL, ?)',
                           (identifier, label, _digest(token), created, expires_at))
            else:
                if db.execute('SELECT 1 FROM demo_api_tokens WHERE digest = ?', (_digest(token),)).fetchone():
                    raise ValueError('Token already exists')
                db.execute('INSERT INTO api_tokens VALUES (?, ?, ?, ?, NULL)',
                           (identifier, label, _digest(token), created))
        return {'id': identifier, 'label': label, 'created_at': created, 'scope': scope, 'expires_at': expires_at, 'token': token}

    def list(self) -> list[dict]:
        with self._connect() as db:
            return [dict(row) for row in db.execute('''
                SELECT id, label, created_at, revoked_at, 'full' AS scope, NULL AS expires_at FROM api_tokens
                UNION ALL
                SELECT id, label, created_at, revoked_at, 'demo' AS scope, expires_at FROM demo_api_tokens
                ORDER BY created_at DESC''')]

    def revoke(self, identifier: str) -> bool:
        with self._connect() as db:
            result = db.execute('UPDATE api_tokens SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL',
                                (_now(), identifier))
            if result.rowcount == 1:
                return True
            result = db.execute('UPDATE demo_api_tokens SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL',
                                (_now(), identifier))
            return result.rowcount == 1

    def valid(self, token: str) -> bool:
        return self.scope(token) is not None

    def scope(self, token: str) -> str | None:
        if not token or len(token) > 512:
            return None
        with self._connect() as db:
            digest = _digest(token)
            demo = db.execute('SELECT revoked_at, expires_at FROM demo_api_tokens WHERE digest = ?', (digest,)).fetchone()
            if demo is not None:
                expires = _deadline(demo['expires_at'])
                return 'demo' if not demo['revoked_at'] and expires is not None and datetime.now(timezone.utc) < expires else None
            full = db.execute('SELECT 1 FROM api_tokens WHERE digest = ? AND revoked_at IS NULL', (digest,)).fetchone()
            return 'full' if full is not None else None


def demo_allowed(path: str, method: str) -> bool:
    """Keep demo access explicit and deny new application features by default."""
    if path == '/api/v1/mcp' or path.startswith('/api/v1/mcp/'):
        return method in {'GET', 'POST', 'DELETE'}
    return (method == 'GET' and path in {
        '/api/v1/status', '/api/v1/graph/overview', '/api/v1/graph/all',
        '/api/v1/graph/search', '/api/v1/graph/neighborhood',
    }) or (method == 'POST' and path in {'/api/v1/query', '/api/v1/query/stream'})


DEMO_PAGES = frozenset({
    '/', '/home', '/RAG', '/graph', '/docs', '/docs/design', '/docs/mcp',
    '/docs/api', '/docs/images/cdm-rag-readme-tour.svg', '/openapi.json',
})


class DemoLease:
    """Keep the admission slot until both request and actual workers finish."""
    def __init__(self, slot: threading.Lock):
        self.slot = slot
        self.guard = threading.Lock()
        self.workers = 0
        self.request_finished = False

    def start_worker(self) -> None:
        with self.guard:
            if self.request_finished:
                raise RuntimeError('Demo request has ended')
            self.workers += 1

    def worker_finished(self, _future) -> None:
        with self.guard:
            self.workers -= 1
            if self.request_finished and not self.workers:
                self.slot.release()

    def finish_request(self) -> None:
        with self.guard:
            self.request_finished = True
            if not self.workers:
                self.slot.release()


_demo_lease: contextvars.ContextVar[DemoLease | None] = contextvars.ContextVar('cdm_demo_lease', default=None)
_demo_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cdm-demo')


async def run_query_worker(func, /, *args, **kwargs):
    """Run a query, attaching demo reservations to the real worker future.

    Cancelling the asyncio wrapper does not terminate a running thread. A
    concurrent Future's done callback fires only when that thread really ends.
    """
    lease = _demo_lease.get()
    if lease is None:
        return await asyncio.to_thread(func, *args, **kwargs)
    lease.start_worker()
    try:
        call = functools.partial(func, *args, **kwargs)
        future = _demo_executor.submit(contextvars.copy_context().run, call)
        future.add_done_callback(lease.worker_finished)
    except BaseException:
        lease.worker_finished(None)
        raise
    return await asyncio.wrap_future(future)


class APIAuthMiddleware:
    """Require a valid bearer token or an authenticated, trusted owner session."""
    def __init__(self, app, *, required: bool, registry: TokenRegistry | None, proxy_secret: str, owner_email: str,
                 owner_subject: str = '', demo_expires_at: str = ''):
        self.app, self.required, self.registry, self.proxy_secret = app, required, registry, proxy_secret
        self.owner_email = owner_email.strip().casefold()
        self.owner_subject = owner_subject.strip()
        self.demo_expires_at = _deadline(demo_expires_at)
        # This independent deployment runs one app worker. These conservative,
        # process-local limits bound a shared demo credential's GPU workload;
        # distributed deployments need a shared rate limiter at the edge.
        self.demo_requests = deque()
        self.demo_queries = deque()
        self.demo_mcp = deque()
        self.demo_slot = threading.Lock()
        if required and (registry is None or len(proxy_secret) < 32 or not self.owner_email or not self.owner_subject):
            raise ValueError('API authentication requires a token registry, a strong proxy secret, and owner identity.')

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not self.required:
            return await self.app(scope, receive, send)
        path = scope['path']
        api = path == '/api' or path.startswith('/api/')
        headers = {k.lower(): v for k, v in scope.get('headers', [])}
        raw_session = headers.get(b'x-cdm-rag-owner-session', b'')
        session = hmac.compare_digest(raw_session, self.proxy_secret.encode('ascii'))
        authorization = headers.get(b'authorization', b'').decode('latin1')
        prefix, sep, credential = authorization.partition(' ')
        token_scope = self.registry.scope(credential) if sep and prefix.lower() == 'bearer' else None
        bearer = token_scope is not None
        management = path == '/api/v1/api-tokens' or path.startswith('/api/v1/api-tokens/')
        mcp = path == '/api/v1/mcp' or path.startswith('/api/v1/mcp/')
        session_email = headers.get(b'x-cdm-rag-owner-email', b'').decode('latin1').strip().casefold()
        session_subject = headers.get(b'x-cdm-rag-owner-subject', b'').decode('latin1').strip()
        is_owner = (session and hmac.compare_digest(session_email.encode(), self.owner_email.encode())
                    and hmac.compare_digest(session_subject.encode(), self.owner_subject.encode()))
        valid_session = session and bool(session_email) and bool(session_subject)
        if api and ((mcp and not bearer) or (management and not is_owner) or (not management and not mcp and not (valid_session or bearer))):
            payload = json.dumps({'detail': 'Authentication required'}).encode()
            await send({'type': 'http.response.start', 'status': 401, 'headers': [(b'content-type', b'application/json'), (b'cache-control', b'no-store'), (b'www-authenticate', b'Bearer')]})
            await send({'type': 'http.response.body', 'body': payload})
            return
        # A bearer credential controls API access even if a session header is
        # also present. Only a verified owner session enables management.
        demo = (api and (token_scope == 'demo' if bearer else valid_session and not is_owner)) or (not api and valid_session and not is_owner)
        if demo and (self.demo_expires_at is None or datetime.now(timezone.utc) >= self.demo_expires_at):
            await send({'type': 'http.response.start', 'status': 403, 'headers': [(b'content-type', b'application/json'), (b'cache-control', b'no-store')]})
            await send({'type': 'http.response.body', 'body': b'{"detail":"Demo access is unavailable"}'})
            return
        if demo and ((api and not demo_allowed(path, scope['method'])) or (not api and (scope['method'] != 'GET' or path not in DEMO_PAGES))):
            payload = json.dumps({'detail': 'Not available to demo accounts'}).encode()
            await send({'type': 'http.response.start', 'status': 403, 'headers': [(b'content-type', b'application/json'), (b'cache-control', b'no-store')]})
            await send({'type': 'http.response.body', 'body': payload})
            return
        reserve_demo = demo and api and scope['method'] == 'POST' and (
            path in {'/api/v1/query', '/api/v1/query/stream'} or mcp)
        lease = DemoLease(self.demo_slot) if reserve_demo and self.demo_slot.acquire(blocking=False) else None
        if reserve_demo and lease is None:
            await send({'type': 'http.response.start', 'status': 429, 'headers': [(b'content-type', b'application/json'), (b'cache-control', b'no-store'), (b'retry-after', b'10')]})
            await send({'type': 'http.response.body', 'body': b'{"detail":"Another demo request is in progress"}'})
            return
        if demo and api:
            now = time.monotonic()
            buckets = [(self.demo_requests, 60, 120)]
            if scope['method'] == 'POST' and path in {'/api/v1/query', '/api/v1/query/stream'}:
                buckets.append((self.demo_queries, 3600, 12))
            if scope['method'] == 'POST' and mcp:
                # MCP handshakes and calls all count: do not inspect JSON-RPC
                # bodies or permit bypass by changing tool arguments.
                buckets.append((self.demo_mcp, 3600, 36))
            for entries, window, limit in buckets:
                while entries and now - entries[0] >= window:
                    entries.popleft()
                if len(entries) >= limit:
                    retry = max(1, int(window - (now - entries[0])) + 1)
                    if lease is not None:
                        lease.finish_request()
                    await send({'type': 'http.response.start', 'status': 429, 'headers': [(b'content-type', b'application/json'), (b'cache-control', b'no-store'), (b'retry-after', str(retry).encode())]})
                    await send({'type': 'http.response.body', 'body': b'{"detail":"Demo request limit reached"}'})
                    return
            for entries, _, _ in buckets:
                entries.append(now)
        scope.setdefault('state', {})['demo'] = bool(demo)
        lease_token = _demo_lease.set(lease) if lease is not None else None
        try:
            return await self.app(scope, receive, send)
        finally:
            if lease is not None:
                _demo_lease.reset(lease_token)
                lease.finish_request()
