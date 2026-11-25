"""End-to-end contracts for clients consuming streamed schema answers."""

import io
import json
import urllib.error

import pytest
from fastapi.testclient import TestClient

from app import main
from scripts import ask_cdm


class StreamingResponse:
    def __init__(self, payload):
        self.lines = io.BytesIO(payload)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.lines.close()

    def readline(self, size):
        return self.lines.readline(size)


def fake_stream(monkeypatch, payload):
    class Opener:
        def open(self, request, timeout):
            return StreamingResponse(payload)

    monkeypatch.setattr(ask_cdm.urllib.request, 'build_opener', lambda *_: Opener())


def test_stream_combines_multiline_events_and_ignores_provisional_tokens(monkeypatch):
    # SSE permits more than one data line in an event; a token is only a draft.
    fake_stream(monkeypatch, (
        ': keep-alive\r\n\r\n'
        'data: {"type":"status",\r\n'
        'data: "stage":"searching_catalog"}\r\n\r\n'
        'data: {"type":"token","token":"provisional"}\r\n\r\n'
        'data: {"type":"result","result":{"answer":"Final [R1]",\r\n'
        'data: "sources":[{"citation":"R1"}],"processing":{"elapsed_ms":12}}}\r\n\r\n'
    ).encode())
    progress = []
    result = ask_cdm.ask_stream('example.test', 'test-token', 'Question', 45, progress.append)
    assert result['answer'] == 'Final [R1]'
    assert result['sources'] == [{'citation': 'R1'}]
    assert result['processing']['elapsed_ms'] == 12
    assert [event['stage'] for event in progress] == ['connection_alive', 'searching_catalog']
    assert all(event.get('type') != 'token' for event in progress)


@pytest.mark.parametrize('event', [
    'data: []\n\n',
    'data: {"type":"result","result":null}\n\n',
    'data: {"type":"result","result":{"answer":null}}\n\n',
    'data: not-json\n\n',
])
def test_malformed_events_are_not_accepted_as_answers(monkeypatch, event):
    fake_stream(monkeypatch, event.encode())
    with pytest.raises(ValueError):
        ask_cdm.ask_stream('example.test', 'test-token', 'Question', 15)


def test_stream_rejects_oversized_event_before_receiving_an_answer(monkeypatch):
    fake_stream(monkeypatch, b'data: ' + b'x' * (4 * 1024 * 1024) + b'\n\n')
    with pytest.raises(ValueError, match='unexpectedly large event'):
        ask_cdm.ask_stream('example.test', 'test-token', 'Question', 15)


def test_server_error_is_not_reported_as_a_success_or_leaked_by_cli(monkeypatch, capsys):
    fake_stream(monkeypatch, b'data: {"type":"error","detail":"private internal failure"}\n\n')
    monkeypatch.setenv('CDM_RAG_GPU_PUBLIC_HOST', 'example.test')
    monkeypatch.setenv('CDM_RAG_API_TOKEN', 'private-test-token')
    monkeypatch.delenv('CDM_RAG_GPU_INITIAL_API_TOKEN', raising=False)
    assert ask_cdm.main(['Question']) == 3
    output = capsys.readouterr()
    assert output.out == ''
    assert 'could not complete' in output.err
    assert 'private internal failure' not in output.err
    assert 'private-test-token' not in output.err


def test_missing_token_is_rejected_without_a_network_request(monkeypatch, capsys):
    monkeypatch.setenv('CDM_RAG_GPU_PUBLIC_HOST', 'example.test')
    monkeypatch.delenv('CDM_RAG_API_TOKEN', raising=False)
    monkeypatch.delenv('CDM_RAG_GPU_INITIAL_API_TOKEN', raising=False)
    monkeypatch.setattr(ask_cdm, 'ask_stream', lambda *_a, **_kw: pytest.fail('Unexpected network request'))
    assert ask_cdm.main(['Question']) == 2
    assert 'CDM_RAG_GPU_INITIAL_API_TOKEN' in capsys.readouterr().err


def test_connection_failure_does_not_echo_network_details_or_credentials(monkeypatch, capsys):
    monkeypatch.setenv('CDM_RAG_GPU_PUBLIC_HOST', 'example.test')
    monkeypatch.setenv('CDM_RAG_API_TOKEN', 'private-test-token')
    monkeypatch.delenv('CDM_RAG_GPU_INITIAL_API_TOKEN', raising=False)

    def fail(*_args, **_kwargs):
        raise urllib.error.URLError('private network detail')

    monkeypatch.setattr(ask_cdm, 'ask_stream', fail)
    assert ask_cdm.main(['Question']) == 3
    output = capsys.readouterr()
    assert output.out == ''
    assert 'Could not reach the server securely' in output.err
    assert 'private network detail' not in output.err
    assert 'private-test-token' not in output.err


def test_stream_failure_emits_a_terminal_error_with_matching_query_id(monkeypatch):
    def fail(*_args, **kwargs):
        kwargs['on_stage']('searching_catalog')
        kwargs['on_token']('unfinished draft')
        raise RuntimeError('private error details')

    monkeypatch.setattr(main, '_run_logged_query', fail)
    response = TestClient(main.app).post('/api/v1/query/stream', json={'question': 'What is Account?'})
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/event-stream')
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert events[0]['stage'] == 'waiting_for_model'
    assert any(event.get('type') == 'token' and event.get('provisional') is True for event in events)
    assert events[-1]['type'] == 'error'
    assert events[-1]['query_id'] == response.headers['x-query-id']
    assert events[-1]['detail'] == 'The query could not be completed.'
    assert all(event.get('type') != 'result' for event in events)
    assert 'private error details' not in response.text
