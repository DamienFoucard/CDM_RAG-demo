import io
import json
import urllib.error

import pytest

from scripts import check_api_token as check


class Response:
    def __init__(self, payload):
        self.status = 200
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return self.payload


def test_explicit_env_file_parses_only_relevant_settings_without_shell_evaluation(tmp_path):
    env_file = tmp_path / 'settings.env'
    env_file.write_text("IGNORED='unterminated\nCDM_RAG_GPU_PUBLIC_HOST='example.test'\n"
                        "CDM_RAG_GPU_INITIAL_API_TOKEN='literal#$(no-command)'\n")
    assert check.read_settings(env_file) == {
        'CDM_RAG_GPU_PUBLIC_HOST': 'example.test',
        'CDM_RAG_GPU_INITIAL_API_TOKEN': 'literal#$(no-command)',
    }


def test_success_sends_only_bearer_header_to_status_endpoint(monkeypatch, capsys):
    token = 'test-token-secret'
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == 'https://example.test/api/v1/status'
            assert request.get_header('Authorization') == 'Bearer ' + token
            assert timeout == 15.0
            return Response(json.dumps({'entity_count': 1, 'passage_count': 2, 'source_commit': 'abc'}).encode())
    monkeypatch.setattr(check.urllib.request, 'build_opener', lambda *args: Opener())
    monkeypatch.setenv('CDM_RAG_GPU_PUBLIC_HOST', 'example.test')
    monkeypatch.setenv('CDM_RAG_GPU_INITIAL_API_TOKEN', token)
    monkeypatch.delenv('CDM_RAG_API_TOKEN', raising=False)
    assert check.main([]) == 0
    captured = capsys.readouterr()
    assert 'HTTP 200' in captured.out
    assert token not in captured.out + captured.err


@pytest.mark.parametrize(('status', 'expected'), [(401, 1), (403, 1), (302, 3), (500, 3)])
def test_http_failures_are_classified_without_printing_secrets(monkeypatch, capsys, status, expected):
    token = 'test-token-secret'
    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, status, 'error', {}, io.BytesIO())
    monkeypatch.setattr(check.urllib.request, 'build_opener', lambda *args: Opener())
    monkeypatch.setenv('CDM_RAG_GPU_PUBLIC_HOST', 'example.test')
    monkeypatch.setenv('CDM_RAG_GPU_INITIAL_API_TOKEN', token)
    monkeypatch.delenv('CDM_RAG_API_TOKEN', raising=False)
    assert check.main([]) == expected
    captured = capsys.readouterr()
    assert str(status) in captured.err
    assert token not in captured.out + captured.err
    assert check.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://other.test') is None


def test_unexpected_html_and_network_failure_are_not_success(monkeypatch):
    class Opener:
        def __init__(self, payload):
            self.payload = payload
        def open(self, request, timeout):
            if self.payload is None:
                raise urllib.error.URLError('network unavailable')
            return Response(self.payload)
    monkeypatch.setattr(check.urllib.request, 'build_opener', lambda *args: Opener(b'<html>Sign in</html>'))
    assert check.check_token('example.test', 'secret', 15)[0] == 3
    monkeypatch.setattr(check.urllib.request, 'build_opener', lambda *args: Opener(None))
    assert check.check_token('example.test', 'secret', 15)[0] == 3


@pytest.mark.parametrize('host', ['http://example.test', 'https://user:pass@example.test',
                                'https://example.test/a', 'https://example.test?x=1'])
def test_rejects_unsafe_or_malformed_origins(host):
    assert check.check_token(host, 'secret', 15)[0] == 2


def test_explicit_file_and_environment_override(monkeypatch, tmp_path):
    env_file = tmp_path / '.env'
    env_file.write_text('CDM_RAG_GPU_PUBLIC_HOST=file.test\nCDM_RAG_GPU_INITIAL_API_TOKEN=file-token\n')
    monkeypatch.setenv('CDM_RAG_GPU_PUBLIC_HOST', 'env.test')
    monkeypatch.setenv('CDM_RAG_API_TOKEN', 'env-token')
    monkeypatch.delenv('CDM_RAG_GPU_INITIAL_API_TOKEN', raising=False)
    calls = []
    monkeypatch.setattr(check, 'check_token', lambda *args: (calls.append(args) or (0, 'accepted')))
    assert check.main(['--env-file', str(env_file), '--timeout', '7']) == 0
    assert calls == [('env.test', 'env-token', 7)]
