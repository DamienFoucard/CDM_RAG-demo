#!/usr/bin/env python3
"""Replay an adaptive router over saved first-pass hits with explicit reasoning.

Model receives the exact saved router messages; gold labels are only used after
inference for scoring.  Run alone: this CLI does not acquire a host model lease.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.adaptive_retrieval_benchmark import ROUTER_SCHEMA, _score, fuse_passages, load_adaptive_retrieval_dataset, routing_messages
from app.retrieval.index import SchemaIndex
from app.runtime.ollama import ModelError, chat_context_size
from app.evaluation.retrieval_benchmark import _slug
from scripts.comparison_ollama import ComparisonOllama, add_comparison_options, comparison_settings


def load_completed(path: Path) -> dict[str, Any]:
    events = [json.loads(line[6:]) for line in path.read_text().splitlines() if line.startswith('data: ')]
    completed = [event['result'] for event in events if event.get('type') == 'retrieval_benchmark_completed']
    if len(completed) != 1:
        raise ValueError('Expected exactly one completed saved retrieval run')
    return completed[0]


def validate_decision(decision: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(decision, dict) or set(decision) != {'search_again', 'query'}:
        raise ValueError('Invalid router decision')
    again, query = decision['search_again'], decision['query']
    if not isinstance(again, bool) or not isinstance(query, str) or len(query) > 2000:
        raise ValueError('Invalid router decision')
    query = ' '.join(query.split())
    if again != bool(query):
        raise ValueError('Inconsistent router decision')
    return {'search_again': again, 'query': query}


def save(path: Path, result: dict[str, Any]) -> None:
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path, help='Completed 9B adaptive SSE run')
    parser.add_argument('--compare-baseline', type=Path, help='Optional paired 12B SSE run for the same split')
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory containing copied retrieval_benchmark_indices')
    parser.add_argument('--model', required=True)
    parser.add_argument('--case-id', action='append', help='Optional case selection; defaults to all saved cases')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    add_comparison_options(parser)
    args = parser.parse_args()
    if args.reasoning_effort != 'on':
        parser.error('This reasoning experiment requires --reasoning-effort on')
    output_limit = args.max_output_tokens or 160
    context_cap = args.context_cap or 8192
    if output_limit >= context_cap:
        parser.error('The output budget must be smaller than the context cap')
    dataset_path = ROOT / 'evaluation/reformulation-challenge-candidate-v1.json'
    dataset = load_adaptive_retrieval_dataset(dataset_path)
    base = load_completed(args.baseline)
    other = load_completed(args.compare_baseline) if args.compare_baseline else None
    for run in (base, other):
        if run and (run['dataset_id'] != dataset['dataset_id'] or run['source_commit'] != dataset['source_commit'] or run['split'] != base['split']):
            raise ValueError('Saved run does not match dataset, source revision or split')
    cases = {case['id']: case for case in dataset['questions']}
    baseline_rows = {row['question_id']: row for row in base['rows']}
    compare_rows = {row['question_id']: row for row in other['rows']} if other else {}
    selected = args.case_id or list(baseline_rows)
    if len(selected) != len(set(selected)) or any(q not in baseline_rows or q not in cases for q in selected):
        raise ValueError('Unknown or duplicate case ID')
    if other and any(q not in compare_rows or compare_rows[q]['first_pass_ids'] != baseline_rows[q]['first_pass_ids'] for q in selected):
        raise ValueError('Comparison run first-pass ranks differ')
    embedding = base['model_names'][0]
    response = httpx.get(args.base_url.rstrip('/') + '/api/tags', timeout=10, trust_env=False)
    response.raise_for_status()
    installed = {item['name']: item for item in response.json()['models']}
    if args.model not in installed or embedding not in installed:
        raise ValueError('Router and embedding models must be installed using their exact tags')
    path = args.data_dir / 'retrieval_benchmark_indices' / _slug(embedding)
    if not (path / 'chroma').exists():
        raise ValueError('Cached retrieval index not found; copy the prior index before running')
    client = ComparisonOllama(args.base_url, embedding, args.model, '__unused_answer__',
                              judge_model='__unused_judge__', request_timeout=600, first_token_timeout=600,
                              reasoning_effort=args.reasoning_effort, max_output_tokens=output_limit, context_cap=context_cap)
    client.query_embedding_keep_alive = '0'
    index = SchemaIndex(ROOT / 'seed/catalog.json', path, client)
    expected_index = base['summaries'][0]['index_identity']
    if not index.index_reused or index.index_revision != expected_index or index.catalog['source_commit'] != dataset['source_commit']:
        raise ValueError('Cached index identity or source revision differs from saved baseline')
    if other and other['summaries'][0]['index_identity'] != expected_index:
        raise ValueError('Comparison index identity differs')
    for q in selected:
        row, case = baseline_rows[q], cases[q]
        if case['split'] != base['split'] or case['question'] != row['question']:
            raise ValueError('Saved question or split changed')
        first = [index.by_id[pid] for pid in row['first_pass_ids']]
        if routing_messages(case['question'], case.get('context_messages') or [], first) != row['router_messages']:
            raise ValueError('Saved router messages do not match reconstructed catalog passages')
        if _score(case, first)['recall_at_6'] != row['variants']['first_pass']['recall_at_6']:
            raise ValueError('Saved first-pass score did not reproduce')
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600); os.close(fd)
    settings = comparison_settings(args.reasoning_effort, output_limit, 160, context_cap)
    result = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'method': 'Replay exact saved first-pass IDs and router messages, then search once and apply existing reciprocal-rank fusion; gold withheld until scoring.',
              'caveat': 'Published source-derived cases, incomplete labels, different reasoning/output settings than saved baselines. Character-based context estimate is not tokenizer proof.',
              'baseline_sha256': hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
              'compare_baseline_sha256': hashlib.sha256(args.compare_baseline.read_bytes()).hexdigest() if other else None,
              'dataset_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'settings': settings, 'model': installed[args.model], 'embedding': installed[embedding],
              'index_identity': index.index_revision, 'source_commit': dataset['source_commit'],
              'split': base['split'], 'rows': []}
    save(args.output, result)
    try:
        for q in selected:
            print(f'{args.model}: {q}', flush=True)
            baseline, case = baseline_rows[q], cases[q]
            first = [index.by_id[pid] for pid in baseline['first_pass_ids']]
            messages = baseline['router_messages']
            row = {'question_id': q, 'question': case['question'], 'first_pass_ids': baseline['first_pass_ids'],
                   'messages_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                   'requested_context_tokens': chat_context_size(messages, output_limit, context_cap),
                   'baseline_9b': baseline['variants']['adaptive'],
                   'baseline_12b': compare_rows[q]['variants']['adaptive'] if other else None}
            started = time.perf_counter()
            try:
                decision, timing = client.chat_json_timed(args.model, messages, schema=ROUTER_SCHEMA, max_tokens=output_limit)
                row['decision'] = validate_decision(decision)
                row['runtime'] = client.runtime_info(args.model)
                row['timing'] = timing
                row['status'] = 'routed'
            except (ModelError, ValueError) as exc:
                row.update(status='error', error=f'{type(exc).__name__}: {exc}')
            row['router_seconds'] = round(time.perf_counter() - started, 3)
            try:
                client.unload_if_loaded(args.model)
            except ModelError as exc:
                row['unload_error'] = str(exc)
                result['rows'].append(row); save(args.output, result)
                raise RuntimeError('Model could not be unloaded; follow-up embedding is unsafe') from exc
            if row['status'] == 'routed':
                second = index.search(row['decision']['query'], limit=20, mode='hybrid') if row['decision']['search_again'] else []
                selected_hits = fuse_passages(first, second) if second else first[:20]
                row.update(first_pass=_score(case, first), adaptive=_score(case, selected_hits),
                           followup_ids=[item['id'] for item in second], fused_ids=[item['id'] for item in selected_hits])
            result['rows'].append(row); save(args.output, result)
        result['status'] = 'completed_with_errors' if any(row['status'] != 'routed' for row in result['rows']) else 'completed'
    except Exception as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            client.unload_if_loaded(args.model)
            client.unload_if_loaded(embedding)
        finally:
            result['completed_at'] = datetime.now(timezone.utc).isoformat(); save(args.output, result)


if __name__ == '__main__':
    main()
