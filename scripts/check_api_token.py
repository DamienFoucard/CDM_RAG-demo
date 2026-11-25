#!/usr/bin/env python3
"""Check a CDM RAG API token without printing it or saving a query."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read_settings(path: Path) -> dict[str, str]:
    """Read only the settings used by this command, without shell evaluation."""
    wanted = {'CDM_RAG_GPU_PUBLIC_HOST', 'CDM_RAG_GPU_INITIAL_API_TOKEN', 'CDM_RAG_API_TOKEN'}
    settings = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        if key.startswith('export '):
            key = key[7:].strip()
        if key not in wanted:
            continue
        parts = shlex.split(value, comments=True)
        if len(parts) != 1:
            raise ValueError(f'Invalid value for {key} in the environment file.')
        if key in settings:
            raise ValueError(f'Duplicate setting {key} in the environment file.')
        settings[key] = parts[0]
    return settings


def status_url(host: str) -> str:
    value = host.strip()
    if '://' not in value:
        value = 'https://' + value
    try:
        parts = urllib.parse.urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise ValueError('The host has an invalid port.') from exc
    if (parts.scheme != 'https' or not parts.hostname or parts.username is not None
            or parts.password is not None or parts.query or parts.fragment
            or parts.path not in ('', '/') or any(ord(c) <= 32 or ord(c) == 127 for c in value)
            or (port is not None and not 1 <= port <= 65535)):
        raise ValueError('Use an HTTPS origin such as https://cdm-rag.example.com.')
    return urllib.parse.urlunsplit(('https', parts.netloc, '/api/v1/status', '', ''))


def check_token(host: str, token: str, timeout: float) -> tuple[int, str]:
    try:
        url = status_url(host)
    except ValueError as exc:
        return 2, str(exc)
    if not token or any(ord(c) < 33 or ord(c) > 126 for c in token):
        return 2, 'Provide a nonempty token without whitespace or control characters.'
    request = urllib.request.Request(url, headers={
        'Authorization': 'Bearer ' + token,
        'Accept': 'application/json',
        'User-Agent': 'CDM-RAG-Token-Check/1.0',
    })
    try:
        # Honour an explicitly configured trust store. Otherwise use certifi when
        # installed (it is bundled with the project dependencies), falling back
        # to Python's default certificate store. Certificate checks stay enabled.
        ca_file = None
        if not (os.environ.get('SSL_CERT_FILE') or os.environ.get('SSL_CERT_DIR')):
            try:
                import certifi
                ca_file = certifi.where()
            except ImportError:
                pass
        context = ssl.create_default_context(cafile=ca_file)
        opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=context))
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                return 3, f'Unexpected HTTP status {response.status}.'
            body = response.read(65537)
        if len(body) > 65536:
            return 3, 'The server returned an unexpectedly large response.'
        result = json.loads(body)
        if not isinstance(result, dict) or not all(k in result for k in ('entity_count', 'passage_count', 'source_commit')):
            return 3, 'The server returned HTTP 200, but not a CDM RAG status response.'
        return 0, 'Token accepted by CDM RAG (HTTP 200).'
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return 1, f'Token rejected or access denied (HTTP {exc.code}).'
        if 300 <= exc.code < 400:
            return 3, f'The server redirected the request (HTTP {exc.code}); no redirect was followed.'
        return 3, f'The server returned HTTP {exc.code}.'
    except (urllib.error.URLError, OSError, TimeoutError):
        return 3, 'Could not reach the server securely, or the request timed out.'
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 3, 'The server did not return a valid CDM RAG status response.'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, help='Read settings from this explicit .env file; environment variables take precedence.')
    parser.add_argument('--url', help='Override the HTTPS host from CDM_RAG_GPU_PUBLIC_HOST.')
    parser.add_argument('--timeout', type=float, default=15.0, help='Request timeout in seconds (default: 15).')
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 300:
        parser.error('--timeout must be greater than 0 and at most 300 seconds')
    try:
        file_settings = read_settings(args.env_file) if args.env_file else {}
    except (OSError, UnicodeError, ValueError) as exc:
        print(f'Could not read the environment file: {type(exc).__name__}.', file=sys.stderr)
        return 2
    host = args.url or os.environ.get('CDM_RAG_GPU_PUBLIC_HOST') or file_settings.get('CDM_RAG_GPU_PUBLIC_HOST', '')
    token = (os.environ.get('CDM_RAG_API_TOKEN') or os.environ.get('CDM_RAG_GPU_INITIAL_API_TOKEN')
             or file_settings.get('CDM_RAG_API_TOKEN') or file_settings.get('CDM_RAG_GPU_INITIAL_API_TOKEN', ''))
    if not host or not token:
        print('Set CDM_RAG_GPU_PUBLIC_HOST and CDM_RAG_GPU_INITIAL_API_TOKEN (or CDM_RAG_API_TOKEN), or pass --env-file.', file=sys.stderr)
        return 2
    code, message = check_token(host, token, args.timeout)
    print(message, file=sys.stdout if code == 0 else sys.stderr)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
