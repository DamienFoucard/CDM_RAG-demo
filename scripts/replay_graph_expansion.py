#!/usr/bin/env python3
"""Replay an experimental, model-free graph expansion on saved adaptive runs.

The expansion only sees the question and candidate passages. Gold IDs are used
solely afterward to score the saved ranking and expanded ranking.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schema.corpus import load_catalog, render_chunks
from app.schema.graph import SchemaGraph

STOP = set('a an and are as at be both by can does for from how in is it its of on or that the their them these this to what where which with would identify find schema model links link relationship relationships references evidence relevant specific something need needs associated involved bank banks'.split())


def words(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(text)).lower()
    result = set()
    for token in re.findall(r'[a-z0-9]+', text):
        if token in STOP or len(token) < 3:
            continue
        # Small, generic suffix normalization, including own / owned.
        for suffix in ('ing', 'ed', 'es', 's'):
            if token.endswith(suffix) and len(token) >= len(suffix) + 3:
                token = token[:-len(suffix)]
                break
        result.add(token)
    return result


def build_expander(catalog):
    graph = SchemaGraph(catalog['entities'], catalog['source_commit'])
    chunks = {item['id']: item for item in render_chunks(catalog)}
    groups = {}
    for source, entity in graph.entities.items():
        relationships = entity.get('relationships', [])
        passages = []
        parts = [entity['name'], entity.get('description', '')]
        for i, relation in enumerate(relationships, 1):
            pid = f'{source}:relationship:{i}'
            if pid not in chunks:
                continue
            # Only materialize actual resolved schema edges; do not invent links.
            edges = [edge for edge in graph.adjacency[source]
                     if edge['source_id'] == source and edge['attribute'] == relation['attribute']]
            if not edges:
                continue
            passages.append(pid)
            parts.extend((relation['attribute'], relation.get('description', '')))
            for edge in edges:
                target = graph.entities[edge['target_id']]
                parts.extend((target['name'], target.get('description', '')))
        if passages:
            groups[source] = {'passages': passages, 'terms': words(' '.join(parts))}
    freq = Counter(term for group in groups.values() for term in group['terms'])
    weights = {term: math.log(1 + len(groups) / count) for term, count in freq.items()}
    return graph, chunks, groups, weights


def expand(question, ranked, graph, chunks, groups, weights, *, seed_count=6, max_hops=2,
           max_expanded=3, retained_prefix=3, limit=20):
    """Expand at most two explicit-edge hops, insert at most three passages.

    Candidate source groups are ranked by query overlap with the source, its
    relationship descriptions, and the resolved target descriptions. No answer
    labels or human reference queries are available to this function.
    """
    seeds = list(dict.fromkeys(item.get('entity_id') for item in ranked[:seed_count]
                               if item.get('entity_id') in graph.entities))
    visited = set(seeds)
    traversed_edges = 0
    frontier = set(seeds)
    for _ in range(max_hops):
        traversed_edges += sum(len(graph.adjacency[source]) for source in frontier)
        adjacent = {edge['target_id'] if edge['source_id'] == source else edge['source_id']
                    for source in frontier for edge in graph.adjacency[source]}
        frontier = adjacent - visited
        visited.update(frontier)
    query = words(question)
    candidates = []
    for source in sorted(visited):
        group = groups.get(source)
        if not group:
            continue
        overlap = query & group['terms']
        score = sum(weights.get(term, 0) for term in overlap)
        if score:
            candidates.append((source, score, sorted(overlap)))
    candidates.sort(key=lambda x: (-x[1], x[0]))
    selected = []
    for source, score, overlap in candidates:
        for pid in groups[source]['passages']:
            selected.append({'id': pid, 'score': round(score, 4), 'matched_terms': overlap})
            if len(selected) >= max_expanded:
                break
        if len(selected) >= max_expanded:
            break
    chosen = [chunks[item['id']] for item in selected]
    ordered = ranked[:retained_prefix] + chosen + ranked[retained_prefix:]
    result, seen = [], set()
    for item in ordered:
        if item['id'] not in seen:
            result.append(item)
            seen.add(item['id'])
    return result[:limit], {'seeds': seeds, 'expanded': selected,
                            'reachable_nodes': len(visited), 'traversed_edges': traversed_edges,
                            'candidate_groups': len(candidates)}


def score(ids, positives):
    gold = set(positives)
    return ({f'recall_at_{k}': len(gold.intersection(ids[:k])) / len(gold) for k in (6, 20)} |
            {f'all_at_{k}': gold.issubset(ids[:k]) for k in (6, 20)} |
            {f'mrr_at_{k}': next((1 / (i + 1) for i, pid in enumerate(ids[:k]) if pid in gold), 0.0)
             for k in (6, 20)} | {'hit_at_1': bool(ids and ids[0] in gold)})


def run(catalog, dataset, paths):
    if catalog['source_commit'] != dataset['source_commit']:
        raise ValueError('Catalog and dataset source revisions differ')
    build_started = time.perf_counter()
    graph, chunks, groups, weights = build_expander(catalog)
    build_ms = (time.perf_counter() - build_started) * 1000
    cases = {case['id']: case for case in dataset['questions']}
    rows = []
    for path in paths:
        events = [json.loads(line[6:]) for line in path.read_text().splitlines() if line.startswith('data: ')]
        completed = [event['result'] for event in events if event.get('type') == 'retrieval_benchmark_completed']
        if len(completed) != 1:
            raise ValueError(f'{path}: expected exactly one completed run')
        run_record = completed[0]
        if run_record['dataset_id'] != dataset['dataset_id'] or run_record['source_commit'] != dataset['source_commit']:
            raise ValueError(f'{path}: run dataset/revision does not match')
        for row in run_record['rows']:
            case = cases[row['question_id']]
            if case['question'] != row['question'] or case['split'] != run_record['split']:
                raise ValueError('Question or split changed')
            first = [chunks[pid] for pid in row['first_pass_ids']]
            second = [chunks[pid] for pid in row['followup_ids']]
            # Exactly match the original reciprocal-rank fusion.
            scores, order, found = {}, {}, {}
            for ranking in (first, second):
                seen = set()
                for i, item in enumerate(ranking, 1):
                    pid = item['id']
                    if pid in seen:
                        continue
                    seen.add(pid)
                    if pid not in found:
                        found[pid], order[pid] = item, len(order)
                    scores[pid] = scores.get(pid, 0) + 1 / (60 + i)
            baseline = ([found[pid] for pid in sorted(scores, key=lambda x: (-scores[x], order[x]))[:20]]
                        if second else first[:20])
            started = time.perf_counter()
            expanded, detail = expand(case['question'], baseline, graph, chunks, groups, weights)
            detail['expansion_ms'] = round((time.perf_counter() - started) * 1000, 3)
            base_ids, expanded_ids = [x['id'] for x in baseline], [x['id'] for x in expanded]
            baseline_score = score(base_ids, case['positive_passage_ids'])
            for key, value in baseline_score.items():
                if abs(float(value) - float(row['variants']['adaptive'][key])) > 1e-8:
                    raise ValueError(f'Saved baseline did not reproduce for {case["id"]} {key}')
            rows.append({'question_id': case['id'], 'split': case['split'], 'run_id': run_record['run_id'],
                         'question': case['question'], 'gold_ids': case['positive_passage_ids'],
                         'first_pass_ids': row['first_pass_ids'], 'baseline_ids': base_ids, 'expanded_ids': expanded_ids,
                         'first_pass': score(row['first_pass_ids'], case['positive_passage_ids']),
                         'baseline': baseline_score, 'graph_expansion': score(expanded_ids, case['positive_passage_ids']),
                         'expansion_detail': detail})
    if len({row['question_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate question IDs')
    return {'dataset_id': dataset['dataset_id'], 'source_commit': dataset['source_commit'],
            'query_model': run_record['query_model'], 'input_paths': [str(path) for path in paths],
            'graph_build_ms': round(build_ms, 3),
            'method': 'two-hop resolved entity relationships; three inserted passages after rank 3; '
                      'lexical overlap with each source and its relationship target descriptions; '
                      'no labels or model calls', 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'seed/catalog.json')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'evaluation/reformulation-challenge-candidate-v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    result = run(load_catalog(args.catalog), json.loads(args.dataset.read_text()), args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for row in result['rows']:
        print(row['question_id'], row['baseline'], '->', row['graph_expansion'])
        print('   expansion:', ', '.join(x['id'] for x in row['expansion_detail']['expanded']))


if __name__ == '__main__':
    main()
