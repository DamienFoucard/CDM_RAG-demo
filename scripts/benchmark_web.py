#!/usr/bin/env python3
"""Compare retrieval approaches on public-question candidates, with a shared entity-context budget."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.index import SchemaIndex
from app.ollama import ModelError, Ollama
from app.rag import expand_context, rerank, cited_answer
from scripts.benchmark_relationships import (VARIANTS, GraphExplorer, NeighborhoodIndex,
    contexts_and_edges, source_key, train_transe)


def web_cases(dataset):
    cases = []
    for item in dataset['questions']:
        for form, field in [('source', 'source_question'), ('adapted', 'prototype_question')]:
            if form == 'adapted' and item[field] == item['source_question']:
                continue
            cases.append({**item, 'case_id': item['id'] + '-' + form,
                          'form': form, 'question': item[field],
                          'out_of_scope': item['id'] in ('web-08', 'web-09')})
    return cases


def context_result(index, question, candidates, selected, by_triple):
    entities, edge_ids = contexts_and_edges(index, question, candidates, selected, by_triple)
    contexts, sources, omitted = expand_context(index, [{'entity_id': eid} for eid in entities])
    return contexts, sources, edge_ids, omitted


def graph_result(index, ranked, edges, by_triple):
    entities = list(dict.fromkeys(eid for edge_id in ranked for eid in
                    (edges[edge_id]['source_id'], edges[edge_id]['target_id'])))
    contexts, sources, omitted = expand_context(index, [{'entity_id': eid} for eid in entities])
    edge_ids = list(dict.fromkeys(by_triple[(edge['source_entity_id'], edge['attribute'], edge['target_entity_id'])]
                    for context in contexts for edge in context['direct_neighbors']))
    return contexts, sources, edge_ids, omitted


def summarize(case, contexts, edge_ids, by_triple, latency, error=None):
    entities = [context['entity_id'] for context in contexts]
    targets = set(case.get('candidate_entities', []))
    edge_targets = set()
    for string in case.get('candidate_edges', []):
        source, rest = string.split(' --', 1)
        attr, target = rest.split('--> ', 1)
        key = (source, attr, target)
        if key not in by_triple:
            raise ValueError(f'Candidate edge not found: {string}')
        edge_targets.add(by_triple[key])
    return {'entities': entities, 'edge_ids': edge_ids,
            'candidate_entity_recall': len(targets & set(entities)) / len(targets) if targets and not case['out_of_scope'] else None,
            'candidate_edge_recall': len(edge_targets & set(edge_ids)) / len(edge_targets) if edge_targets and not case['out_of_scope'] else None,
            'latency_ms': round(latency, 1), 'error': error}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=ROOT/'evaluation/web-question-candidates-v1.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raw = args.dataset.read_bytes(); data = json.loads(raw); cases = web_cases(data)
    models = Ollama(os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434'),
                    os.getenv('EMBEDDING_MODEL','qwen3-embedding:0.6b'), 'qwen3:4b','qwen3:4b')
    index = SchemaIndex(ROOT/'seed/catalog.json', args.data_dir/'baseline', models)
    enriched = NeighborhoodIndex(index, models, args.data_dir/'neighborhood-chroma')
    edges = index.graph.edges; by_triple = {source_key(edge): edge['id'] for edge in edges.values()}
    trained, transe_info = train_transe(index.graph); explorer = GraphExplorer(index, models, trained)
    warm = index.search(cases[0]['question'], limit=20, mode='hybrid')
    enriched.search(cases[0]['question'], limit=20); rerank(cases[0]['question'], warm, models, 6)
    rows = []
    for case in cases:
        q = case['question']; outputs = {}; contexts_by_variant = {}; sources_by_variant = {}
        start = time.perf_counter(); candidates = index.search(q, limit=20, mode='hybrid')
        search_ms = 1000*(time.perf_counter()-start)
        start = time.perf_counter()
        contexts, sources, eids, _ = context_result(index,q,candidates,candidates[:6],by_triple)
        outputs['hybrid_raw'] = summarize(case,contexts,eids,by_triple,search_ms+1000*(time.perf_counter()-start))
        contexts_by_variant['hybrid_raw'], sources_by_variant['hybrid_raw'] = contexts, sources
        start = time.perf_counter(); error = None
        try:
            selected = rerank(q,candidates,models,6)
            contexts,sources,eids,_ = context_result(index,q,candidates,selected,by_triple)
        except ModelError as exc:
            contexts,sources,eids,error = [],[],[],str(exc)
        baseline_ms = search_ms+1000*(time.perf_counter()-start)
        outputs['hybrid_reranked'] = summarize(case,contexts,eids,by_triple,baseline_ms,error)
        contexts_by_variant['hybrid_reranked'], sources_by_variant['hybrid_reranked'] = contexts,sources
        seeds = [c['entity_id'] for c in contexts]
        start = time.perf_counter(); error = None
        try:
            ecandidates = enriched.search(q, limit=20); selected = rerank(q,ecandidates,models,6)
            contexts,sources,eids,_ = context_result(index,q,ecandidates,selected,by_triple)
        except ModelError as exc:
            contexts,sources,eids,error = [],[],[],str(exc)
        outputs['neighborhood_reranked'] = summarize(case,contexts,eids,by_triple,1000*(time.perf_counter()-start),error)
        contexts_by_variant['neighborhood_reranked'], sources_by_variant['neighborhood_reranked'] = contexts,sources
        start = time.perf_counter(); a,b,info = explorer.rank(q,seeds)
        graph_ms = 1000*(time.perf_counter()-start)
        for name,ranked in [('graph_assisted',a),('graph_transe',b)]:
            contexts,sources,eids,_ = graph_result(index,ranked,edges,by_triple)
            outputs[name] = summarize(case,contexts,eids,by_triple,baseline_ms+graph_ms)
            outputs[name]['ranked_graph_edges'] = ranked
            contexts_by_variant[name], sources_by_variant[name] = contexts,sources
        # For the two clearly unanswerable requests, inspect the same answer generator
        # against the evidence returned by each approach, independently of routing.
        if case['out_of_scope']:
            for name in VARIANTS:
                try:
                    answer, model = cited_answer(q, sources_by_variant[name], contexts_by_variant[name], models)
                    outputs[name]['answer'] = answer
                    outputs[name]['answer_model'] = model
                except ModelError as exc:
                    outputs[name]['answer_error'] = str(exc)
        rows.append({**case, 'variants':outputs, 'graph_search':info})
        partial = {'dataset_id':data['dataset_id'], 'dataset_status':data['status'],
                   'dataset_sha256':hashlib.sha256(raw).hexdigest(), 'source_commit':index.catalog['source_commit'],
                   'embedding_model':models.embedding_model, 'embedding_digest':index.model_digest,
                   'reranker_model':models.answer_model, 'baseline_index_revision':index.index_revision,
                   'neighborhood_index_revision':enriched.identity, 'transe':transe_info,
                   'settings':{'query':'raw question; router excluded', 'candidate_chunks':20,
                               'reranked_chunks':6,'max_complete_entities':4,'max_context_chars':85000,
                               'graph_ranked_edge_limit':20,
                               'graph_expansion':'ordered unique edge endpoints expanded to complete entity contexts'},
                   'cases':rows}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(partial,indent=2,ensure_ascii=False)+'\n')
        print(f"{len(rows)}/{len(cases)} {case['case_id']}",flush=True)
    metrics = {}
    for name in VARIANTS:
        entries = [r['variants'][name] for r in rows]
        entity = [e['candidate_entity_recall'] for e in entries if e['candidate_entity_recall'] is not None]
        edge = [e['candidate_edge_recall'] for e in entries if e['candidate_edge_recall'] is not None]
        metrics[name] = {'cases':len(entries),'candidate_entity_cases':len(entity),'candidate_edge_cases':len(edge),
                         'mean_candidate_entity_recall':statistics.mean(entity),
                         'mean_candidate_edge_recall':statistics.mean(edge),
                         'mean_latency_ms':statistics.mean(e['latency_ms'] for e in entries),
                         'retrieval_errors':sum(bool(e['error']) for e in entries)}
    partial['metrics'] = metrics
    args.output.write_text(json.dumps(partial,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(metrics,indent=2),flush=True)

if __name__ == '__main__': main()
