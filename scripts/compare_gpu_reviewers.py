#!/usr/bin/env python3
"""Replay fixed, previously audited drafts on reviewers, preserving exact requests.

The GPU service must be dedicated to this evaluation while it is running: Ollama's
model residency controls are shared across clients. Do not point this at a shared
instance without arranging a quiet window first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.comparison_ollama import ComparisonOllama
from scripts.review_saved_claims import OBSERVATIONS, load_review_function
from scripts.replay_reviewer_positive_controls import collect_controls


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def store(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        os.fchmod(file.fileno(), 0o600)
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write('\n')
    os.replace(temporary, path)


class RecordingOllama(ComparisonOllama):
    def _chat_payload(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        request = super()._chat_payload(*args, **kwargs)
        self.request = request
        return request

    def chat(self, *args: Any, **kwargs: Any) -> str:
        result = super().chat(*args, **kwargs)
        self.raw_response = result
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, help='Versioned, self-contained case fixture')
    parser.add_argument('--input', type=Path, action='append', help='Original saved paired-answer runs')
    parser.add_argument('--controls', type=Path, help='Original saved answer event stream')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--judge', action='append', required=True)
    parser.add_argument('--case', action='append', help='Select a fixture case by ID; repeat for a targeted diagnostic')
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    parser.add_argument('--context-cap', type=int, default=8192)
    parser.add_argument('--max-output-tokens', type=int, default=2048)
    parser.add_argument('--reasoning-effort', choices=('on', 'low', 'medium', 'high'))
    parser.add_argument('--hardware', default='unspecified')
    args = parser.parse_args()
    if bool(args.fixtures) == bool(args.input or args.controls):
        parser.error('provide either --fixtures or both --input and --controls')
    if not args.fixtures and not (args.input and args.controls):
        parser.error('original inputs require at least one --input and --controls')
    source_hashes = {}
    cases = []
    for path in args.input or []:
        raw = path.read_bytes()
        source_hashes[path.name] = sha(raw)
        for case in json.loads(raw)['rows']:
            for arm_name, arm in case['arms'].items():
                key = (case['id'], arm_name)
                if key in OBSERVATIONS:
                    if arm['status'] != 'scored':
                        raise ValueError(f'Not scored: {key}')
                    cases.append({'id': '/'.join(key), 'expected': 'flag',
                                  'observation': OBSERVATIONS[key], 'question': case['question'],
                                  'answer': arm['raw_answer'], 'sources': arm['sources'], 'contexts': arm['contexts']})
    if args.fixtures:
        raw = args.fixtures.read_bytes()
        source_hashes[args.fixtures.name] = sha(raw)
        cases = json.loads(raw)['cases']
    else:
        if len(cases) != len(OBSERVATIONS) or len({case['id'] for case in cases}) != len(OBSERVATIONS):
            raise ValueError('Expected five unique unsupported-draft examples')
        raw = args.controls.read_bytes()
        source_hashes[args.controls.name] = sha(raw)
        for case in collect_controls(raw):
            cases.append({'id': f"{case['draft_model']}/{case['question_id']}", 'expected': 'accept',
                          'observation': 'Previously reviewed as supported; original graph contexts unavailable.',
                          **{key: case[key] for key in ('question', 'answer', 'sources', 'contexts')}})
    if (len(cases) != 9 or len({case['id'] for case in cases}) != 9
            or sum(case['expected'] == 'flag' for case in cases) != 5
            or sum(case['expected'] == 'accept' for case in cases) != 4):
        raise ValueError('Expected five flagged drafts and four unique supported controls')
    if args.case:
        requested = set(args.case)
        if len(requested) != len(args.case) or not requested <= {case['id'] for case in cases}:
            raise ValueError('Requested case IDs must be unique and present in the fixture')
        cases = [case for case in cases if case['id'] in requested]
    tags = httpx.get(args.base_url + '/api/tags', timeout=10, trust_env=False)
    tags.raise_for_status()
    available = {item['name']: item for item in tags.json()['models']}
    if any(tag not in available for tag in args.judge):
        raise ValueError('All models must already be installed, under exact tags')
    if args.output.exists():
        raise FileExistsError(args.output)
    review, review_hash = load_review_function(None)
    result = {'status': 'running', 'created_at': datetime.now(timezone.utc).isoformat(),
              'hardware': args.hardware, 'inputs_sha256': source_hashes,
              'review_source_sha256': review_hash,
              'revision': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                                         check=True, text=True).stdout.strip(),
              'implementation_sha256': {name: sha((ROOT / name).read_bytes()) for name in (
                  'app/runtime/ollama.py', 'scripts/comparison_ollama.py', 'scripts/compare_gpu_reviewers.py')},
              'settings': {'think': args.reasoning_effort or False, 'max_output_tokens': args.max_output_tokens,
                           'context_cap': args.context_cap, 'temperature': 0},
              'method': 'Subset of five previously identified problematic drafts and four supported controls. '
                        'Observations and expected decisions are withheld from the reviewer. '
                        'Controls lack original graph contexts; two share a question and evidence. '
                        'One pass per model in fixed order; the first call can include model loading. '
                        'Cases are correlated and labels are not independently reviewed. Flagging alone '
                        'does not prove the reviewer identified the manually recorded problem.',
              'selected_case_ids': [case['id'] for case in cases],
              'models': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    store(args.output, result)
    try:
        for tag in args.judge:
            models = RecordingOllama(args.base_url, 'nomic-embed-text:latest', tag, tag, judge_model=tag,
                                     reasoning_effort=args.reasoning_effort,
                                     max_output_tokens=args.max_output_tokens, context_cap=args.context_cap)
            models.unload_other_models(tag)
            group = {'model': available[tag], 'rows': []}
            result['models'].append(group)
            try:
                for case in cases:
                    row = {key: case[key] for key in ('id', 'expected', 'observation')}
                    models.request = None
                    models.raw_response = None
                    started = time.perf_counter()
                    try:
                        reviewed = review(case['question'], case['answer'], case['sources'], case['contexts'], models)
                        row.update(status='reviewed', review=reviewed,
                                   matches_expected=(reviewed['decision'] == 'accept') == (case['expected'] == 'accept'))
                    except Exception as exc:
                        row.update(status='error', error=f'{type(exc).__name__}: {exc}')
                    row['elapsed_seconds'] = round(time.perf_counter() - started, 3)
                    row['runtime'] = models.runtime_info(tag)
                    row['ollama_timings'] = getattr(models._chat_local, 'timings', None)
                    row['request'] = models.request
                    row['raw_response'] = models.raw_response
                    group['rows'].append(row)
                    store(args.output, result)
                    print(f"{tag} {case['id']}: {row['status']} {row.get('review', {}).get('decision', '')} {row['elapsed_seconds']}s", flush=True)
            finally:
                models.unload_if_loaded(tag)
        result['status'] = 'completed_with_errors' if any(row['status'] != 'reviewed' for group in result['models'] for row in group['rows']) else 'completed'
    except BaseException as exc:
        result.update(status='interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        store(args.output, result)


if __name__ == '__main__':
    main()
