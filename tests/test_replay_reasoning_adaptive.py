import json
from pathlib import Path

import pytest

from scripts.replay_reasoning_adaptive import load_completed, validate_decision


def test_saved_run_requires_exactly_one_completion(tmp_path: Path):
    path = tmp_path / 'events.sse'
    path.write_text(': keep-alive\n\ndata: ' + json.dumps({'type': 'retrieval_benchmark_completed', 'result': {'ok': True}}) + '\n')
    assert load_completed(path) == {'ok': True}
    path.write_text('data: {"type":"other"}\n')
    with pytest.raises(ValueError):
        load_completed(path)


def test_router_decisions_reject_inconsistent_or_invalid_search():
    assert validate_decision({'search_again': True, 'query': '  holding   details '}) == {'search_again': True, 'query': 'holding details'}
    assert validate_decision({'search_again': False, 'query': '  '}) == {'search_again': False, 'query': ''}
    for value in ({'search_again': True, 'query': ''}, {'search_again': False, 'query': 'query'},
                  {'search_again': 'true', 'query': 'x'}, {'search_again': True, 'query': 'x', 'gold': 'leak'}):
        with pytest.raises(ValueError):
            validate_decision(value)
