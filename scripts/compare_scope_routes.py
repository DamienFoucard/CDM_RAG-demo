#!/usr/bin/env python3
"""Pair baseline and scope-guard routing decisions on shared question IDs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def summarize(rows: list[dict]) -> dict:
    schema = [row for row in rows if row['expected'] == 'schema']
    other = [row for row in rows if row['expected'] != 'schema']
    correct = sum(row['search'] == (row['expected'] == 'schema') for row in rows)
    return {
        'cases': len(rows),
        'correct_routes': correct,
        'accuracy': round(correct / len(rows), 4) if rows else None,
        'schema_cases': len(schema),
        'schema_searches': sum(row['search'] is True for row in schema),
        'schema_recall': round(sum(row['search'] is True for row in schema) / len(schema), 4) if schema else None,
        'non_schema_cases': len(other),
        'non_schema_searches': sum(row['search'] is True for row in other),
        'non_schema_false_search_rate': round(sum(row['search'] is True for row in other) / len(other), 4) if other else None,
        'errors': sum(row['search'] is None for row in rows),
    }


def pair(base: list[dict], guarded: list[dict], key: str, expected_key: str) -> list[dict]:
    before = {row[key]: row for row in base}
    after = {row[key]: row for row in guarded}
    if before.keys() != after.keys():
        missing = sorted(before.keys() ^ after.keys())
        raise SystemExit(f'Unpaired {key} values: {missing}')
    rows = []
    for item in guarded:
        old, new = before[item[key]], item
        expectation = new[expected_key]
        old_decision = old.get('search')
        if old_decision is None:
            old_decision = old.get('decision') == 'search'
        rows.append({
            key: item[key], 'question': item['question'], 'expected': expectation,
            'baseline_search': old_decision, 'guard_search': new['search'],
            'scope': new.get('scope'),
        })
    return rows


def section(rows: list[dict]) -> dict:
    return {
        'baseline': summarize([{**r, 'search': r['baseline_search']} for r in rows]),
        'guarded': summarize([{**r, 'search': r['guard_search']} for r in rows]),
        'cases': rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, required=True)
    p.add_argument('--web-results-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sd = args.results_dir / 'scope-v1'
    baseline = json.loads((sd / 'baseline.json').read_bytes())
    guarded = json.loads((sd / 'guard-test.json').read_bytes())
    baseline_test = [row for row in baseline['cases'] if row['split'] == 'test']
    synthetic = pair(baseline_test, guarded['cases'], 'id', 'expected')

    web_base = json.loads((args.web_results_dir / 'routing.json').read_bytes())['policies']['planner_4b']
    web_guard = json.loads((sd / 'web-guard.json').read_bytes())
    web_base = [{**row, 'id': row['case_id']} for row in web_base
                if row['case_id'] in {r['id'] for r in web_guard['cases']}]
    web = pair(web_base, web_guard['cases'], 'id', 'expected')
    result = {
        'comparison_id': 'scope-routing-comparison-v1',
        'label_status': 'Assistant-authored and unreviewed; web labels are policy interpretations of public examples.',
        'metrics': {
            'synthetic_baseline_accuracy': section(synthetic)['baseline']['accuracy'],
            'synthetic_guard_accuracy': section(synthetic)['guarded']['accuracy'],
            'synthetic_baseline_non_schema_false_search_rate': section(synthetic)['baseline']['non_schema_false_search_rate'],
            'synthetic_guard_non_schema_false_search_rate': section(synthetic)['guarded']['non_schema_false_search_rate'],
            'web_baseline_accuracy': section(web)['baseline']['accuracy'],
            'web_guard_accuracy': section(web)['guarded']['accuracy'],
            'web_baseline_non_schema_false_search_rate': section(web)['baseline']['non_schema_false_search_rate'],
            'web_guard_non_schema_false_search_rate': section(web)['guarded']['non_schema_false_search_rate'],
        },
        'datasets': {
            'synthetic_dataset_sha256': guarded['dataset_sha256'],
            'web_dataset_sha256': web_guard['dataset_sha256'],
        },
        'comparisons': {
            'held_out_synthetic_test': section(synthetic),
            'web_question_pilot': section(web),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: {policy: value[policy] for policy in ('baseline', 'guarded')}
                      for k, value in result['comparisons'].items()}, indent=2))


if __name__ == '__main__':
    main()
