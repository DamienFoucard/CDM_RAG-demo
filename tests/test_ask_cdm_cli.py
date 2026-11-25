"""Exercise the terminal question client without contacting the live service."""
import io
import json
import urllib.error

import pytest

from scripts import ask_cdm as cli


class Response:
    def __init__(self, payload):
        self.payload = payload
        self.stream = io.BytesIO(payload)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self, limit):
        return self.payload
    def readline(self, limit):
        return self.stream.readline(limit)


def test_question_request_uses_bearer_and_prints_answer_without_token(monkeypatch, tmp_path, capsys):
    token = 'cdmr_test_secret'
    env_file = tmp_path / '.env'
    env_file.write_text(f"CDM_RAG_GPU_PUBLIC_HOST='example.test'\nCDM_RAG_GPU_INITIAL_API_TOKEN='{token}'\n")
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == 'https://example.test/api/v1/query/stream'
            assert request.method == 'POST'
            assert request.get_header('Authorization') == 'Bearer ' + token
            assert json.loads(request.data) == {'question': 'What is Account?'}
            assert timeout == 600
            return Response(b': keep-alive\n\ndata: {"type":"status","stage":"checking_scope"}\n\n'
                            b'data: {"type":"model","role":"scope","model":"scope:1"}\n\n'
                            b'data: {"type":"token","token":"draft"}\n\n'
                            b'data: {"type":"result","result":{"answer":"A cited definition [R1].","sources":[{"id":"R1"}]}}\n\n')
    monkeypatch.setattr(cli.urllib.request, 'build_opener', lambda *args: Opener())
    monkeypatch.delenv('CDM_RAG_GPU_PUBLIC_HOST', raising=False)
    monkeypatch.delenv('CDM_RAG_GPU_INITIAL_API_TOKEN', raising=False)
    monkeypatch.delenv('CDM_RAG_API_TOKEN', raising=False)
    assert cli.main(['--env-file', str(env_file), 'What is Account?']) == 0
    result = capsys.readouterr()
    assert result.out.strip() == 'A cited definition [R1].'
    assert 'Checking scope' in result.err and 'connection active' in result.err
    assert 'Using scope:1 for scope' in result.err
    assert 'draft' not in result.out
    assert token not in result.out + result.err


def test_json_option_includes_sources(monkeypatch, capsys):
    monkeypatch.setenv('CDM_RAG_GPU_PUBLIC_HOST', 'example.test')
    monkeypatch.setenv('CDM_RAG_GPU_INITIAL_API_TOKEN', 'secret')
    monkeypatch.delenv('CDM_RAG_API_TOKEN', raising=False)
    monkeypatch.setattr(cli, 'ask_stream', lambda *args, **kwargs: {'answer': 'A', 'sources': [{'id': 'R1'}]})
    assert cli.main(['--json', 'Question']) == 0
    assert json.loads(capsys.readouterr().out)['sources'] == [{'id': 'R1'}]


@pytest.mark.parametrize(('status', 'exit_code'), [(401, 1), (403, 1), (302, 3), (500, 3)])
def test_http_errors_never_print_token(monkeypatch, capsys, status, exit_code):
    token = 'example-secret'
    monkeypatch.setenv('CDM_RAG_GPU_PUBLIC_HOST', 'example.test')
    monkeypatch.setenv('CDM_RAG_GPU_INITIAL_API_TOKEN', token)
    monkeypatch.delenv('CDM_RAG_API_TOKEN', raising=False)
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError('https://example.test/api/v1/query', status, '', {}, io.BytesIO())
    monkeypatch.setattr(cli, 'ask_stream', fail)
    assert cli.main(['Question']) == exit_code
    output = capsys.readouterr()
    assert str(status) in output.err
    assert token not in output.out + output.err


def test_invalid_origin_and_question_rejected_before_request(monkeypatch):
    monkeypatch.setattr(cli.urllib.request, 'build_opener', lambda *args: pytest.fail('network was accessed'))
    with pytest.raises(ValueError):
        cli.ask('http://example.test', 'token', 'Question', 15)
    with pytest.raises(ValueError):
        cli.ask('example.test', 'token', '   ', 15)


def test_stream_decodes_utf8_and_rejects_truncated_stream(monkeypatch):
    payload = ('data: {"type":"result","result":{"answer":"Grüße"}}\n\n').encode()
    class Opener:
        def open(self, request, timeout):
            return Response(payload)
    monkeypatch.setattr(cli.urllib.request, 'build_opener', lambda *args: Opener())
    assert cli.ask_stream('example.test', 'token', 'Question', 15)['answer'] == 'Grüße'
    payload = b'data: {"type":"status","stage":"checking_scope"}\n\n'
    with pytest.raises(ValueError, match='before its final result'):
        cli.ask_stream('example.test', 'token', 'Question', 15)
