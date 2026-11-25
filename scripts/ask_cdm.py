#!/usr/bin/env python3
"""Ask the CDM RAG API a question using a token from a private environment file."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import ssl
import sys
import urllib.error
import urllib.request
from collections.abc import Callable

if __package__:
    from .check_api_token import NoRedirect, read_settings, status_url
else:
    from check_api_token import NoRedirect, read_settings, status_url


def ask(host: str, token: str, question: str, timeout: float) -> dict:
    if not token or any(ord(c) < 33 or ord(c) > 126 for c in token):
        raise ValueError('Provide a nonempty token without whitespace or control characters.')
    question = ' '.join(question.split())
    if not question or len(question) > 2000:
        raise ValueError('Provide a question between 1 and 2000 characters.')
    url = status_url(host).removesuffix('/status') + '/query'
    data = json.dumps({'question': question}).encode('utf-8')
    request = urllib.request.Request(url, data=data, method='POST', headers={
        'Authorization': 'Bearer ' + token,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': 'CDM-RAG-Question-CLI/1.0',
    })
    ca_file = None
    if not (os.environ.get('SSL_CERT_FILE') or os.environ.get('SSL_CERT_DIR')):
        try:
            import certifi
            ca_file = certifi.where()
        except ImportError:
            pass
    opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(
        context=ssl.create_default_context(cafile=ca_file)))
    with opener.open(request, timeout=timeout) as response:
        body = response.read(4 * 1024 * 1024 + 1)
    if len(body) > 4 * 1024 * 1024:
        raise ValueError('The server returned an unexpectedly large response.')
    result = json.loads(body)
    if not isinstance(result, dict) or not isinstance(result.get('answer'), str):
        raise ValueError('The server returned an invalid answer response.')
    return result


def ask_stream(host: str, token: str, question: str, timeout: float,
               on_status: Callable[[dict], None] | None = None) -> dict:
    """Read SSE incrementally; result.answer, not streamed draft tokens, is final."""
    if not token or any(ord(c) < 33 or ord(c) > 126 for c in token):
        raise ValueError('Provide a nonempty token without whitespace or control characters.')
    question = ' '.join(question.split())
    if not question or len(question) > 2000:
        raise ValueError('Provide a question between 1 and 2000 characters.')
    url = status_url(host).removesuffix('/status') + '/query/stream'
    request = urllib.request.Request(url, data=json.dumps({'question': question}).encode('utf-8'),
                                     method='POST', headers={
        'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json',
        'Accept': 'text/event-stream', 'User-Agent': 'CDM-RAG-Question-CLI/1.0',
    })
    ca_file = None
    if not (os.environ.get('SSL_CERT_FILE') or os.environ.get('SSL_CERT_DIR')):
        try:
            import certifi
            ca_file = certifi.where()
        except ImportError:
            pass
    opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(
        context=ssl.create_default_context(cafile=ca_file)))
    with opener.open(request, timeout=timeout) as response:
        data_lines: list[str] = []
        size = 0
        while True:
            raw = response.readline(4 * 1024 * 1024 + 1)
            if not raw:
                raise ValueError('The stream ended before its final result.')
            size += len(raw)
            if size > 4 * 1024 * 1024:
                raise ValueError('The server returned an unexpectedly large event.')
            line = raw.decode('utf-8').rstrip('\r\n')
            if line.startswith(':'):
                if on_status:
                    on_status({'stage': 'connection_alive'})
            elif line.startswith('data:'):
                data_lines.append(line[5:].lstrip(' '))
            elif not line:
                if data_lines:
                    event = json.loads('\n'.join(data_lines))
                    if not isinstance(event, dict):
                        raise ValueError('Invalid stream event.')
                    if event.get('type') in ('status', 'model') and on_status:
                        on_status(event)
                    elif event.get('type') == 'error':
                        raise RuntimeError('The server could not complete the question.')
                    elif event.get('type') == 'result':
                        result = event.get('result')
                        if not isinstance(result, dict) or not isinstance(result.get('answer'), str):
                            raise ValueError('The server returned an invalid answer response.')
                        return result
                data_lines.clear()
                size = 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question', help='The schema question to ask.')
    parser.add_argument('--env-file', type=Path, help='Read credentials from this explicit environment file; environment variables take precedence.')
    parser.add_argument('--url', help='Override the HTTPS host from CDM_RAG_GPU_PUBLIC_HOST.')
    parser.add_argument('--timeout', type=float, default=600.0, help='Connection inactivity timeout in seconds (default: 600).')
    parser.add_argument('--json', action='store_true', help='Print the complete JSON response including source evidence.')
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 3600:
        parser.error('--timeout must be greater than 0 and at most 3600 seconds')
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
    try:
        def show_status(event: dict) -> None:
            if event.get('type') == 'model':
                role, model = event.get('role'), event.get('model')
                if (isinstance(role, str) and isinstance(model, str)
                        and len(role) < 80 and len(model) < 160
                        and all(c.isprintable() for c in role + model)):
                    print(f'Using {model} for {role.replace("_", " ")}...', file=sys.stderr, flush=True)
                return
            stage = event.get('stage', '')
            if stage == 'connection_alive':
                print('Waiting for the next update (connection active)...', file=sys.stderr, flush=True)
            elif isinstance(stage, str) and stage.replace('_', '').isalnum() and len(stage) < 80:
                print(stage.replace('_', ' ').capitalize() + '...', file=sys.stderr, flush=True)
        result = ask_stream(host, token, args.question, args.timeout, on_status=show_status)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            print(f'Token rejected or access denied (HTTP {exc.code}).', file=sys.stderr)
            return 1
        if 300 <= exc.code < 400:
            print(f'The server redirected the request (HTTP {exc.code}); no redirect was followed.', file=sys.stderr)
        else:
            print(f'The server returned HTTP {exc.code}.', file=sys.stderr)
        return 3
    except (urllib.error.URLError, OSError, TimeoutError):
        print('Could not reach the server securely, or the request timed out.', file=sys.stderr)
        return 3
    except (ValueError, UnicodeError):
        print('Invalid host, token, question or server response.', file=sys.stderr)
        return 2
    except RuntimeError:
        print('The server could not complete the question.', file=sys.stderr)
        return 3
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result['answer'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
