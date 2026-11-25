#!/usr/bin/env python3
"""Paired, fixed-candidate reranking experiment (run with a private CDM index copy)."""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, sys, time
from pathlib import Path
ROOT = Path(os.getenv('CDM_RAG_REPO_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from app.retrieval.index import SchemaIndex
from app.runtime.ollama import Ollama
from app.retrieval.rag import rerank

DEFAULT_DATASET = ROOT / 'evaluation/cdm-schema-evaluation-v2.json'

class BenchmarkOllama(Ollama):
    def __init__(self, *args, reasoning='off', max_output=300, **kwargs):
        super().__init__(*args, **kwargs); self.reasoning=reasoning; self.max_output=max_output
    def chat_json(self, model, messages, *, schema=None, max_tokens=300):
        return super().chat_json(model, messages, schema=schema, max_tokens=self.max_output)
    def _chat_payload(self, model, messages, **kwargs):
        payload=super()._chat_payload(model, messages, **kwargs)
        payload['think']=False if self.reasoning=='off' else self.reasoning
        return payload

def score(passages, gold, target_kind):
    entities = list(dict.fromkeys(p.get('entity_id') for p in passages if p.get('entity_id')))
    ids = [p['id'] for p in passages]
    ranked = entities if target_kind == 'entity' else list(dict.fromkeys(ids))
    rank = next((i for i, value in enumerate(ranked, 1) if value in gold), None)
    return {'entities': entities, 'passage_ids': ids,
            'hit': rank is not None, 'reciprocal_rank': 1 / rank if rank else 0.0,
            'recall': len(set(ranked) & set(gold)) / len(gold)}

def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['snapshot','rank'])
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model')
    parser.add_argument('--reasoning', choices=['off','low'], default='off')
    parser.add_argument('--max-output', type=int, default=300)
    parser.add_argument('--data-dir', type=Path, default=Path('/tmp/cdm-rag-rerank-index'))
    args=parser.parse_args()
    base=os.getenv('OLLAMA_BASE_URL','http://127.0.0.1:11434')
    if args.phase == 'snapshot':
        raw=args.dataset.read_bytes(); dataset=json.loads(raw)
        models=Ollama(base,os.getenv('EMBEDDING_MODEL','qwen3-embedding:0.6b'),'qwen3:4b','qwen3.5:9b')
        index=SchemaIndex(ROOT/'seed/catalog.json', args.data_dir, models)
        if index.catalog['source_commit'] != dataset['source_commit']: raise SystemExit('Catalog mismatch')
        cases=[]
        test_questions=[q for q in dataset['questions'] if q.get('split') == 'test']
        if not test_questions: raise SystemExit('No test questions')
        target_kind='passage' if all('positive_passage_ids' in q for q in test_questions) else 'entity'
        key='positive_passage_ids' if target_kind=='passage' else 'positive_entity_ids'
        if not all(q.get(key) for q in test_questions): raise SystemExit('Missing consistent target labels')
        for q in dataset['questions']:
            if q['split'] != 'test': continue
            start=time.perf_counter(); candidates=index.search(q['question'],limit=20,mode='hybrid')
            cases.append({'id':q['id'],'question':q['question'],'targets':q[key],
                          'candidate_search_ms':round(1000*(time.perf_counter()-start),1),'candidates':candidates})
            print('Snapshot',len(cases),q['id'],flush=True)
        write(args.snapshot,{'dataset':dataset['dataset_id'],'dataset_sha256':hashlib.sha256(raw).hexdigest(),
             'source_commit':dataset['source_commit'],'index_revision':index.index_revision,
             'embedding_model':models.embedding_model,'embedding_digest':index.model_digest,'target_kind':target_kind,'cases':cases})
        return
    if not args.model or not args.output: parser.error('rank requires --model and --output')
    raw=args.snapshot.read_bytes(); snapshot=json.loads(raw)
    models=BenchmarkOllama(base,snapshot['embedding_model'],'qwen3:4b',args.model, reasoning=args.reasoning, max_output=args.max_output)
    result={'dataset':snapshot['dataset'],'dataset_sha256':snapshot['dataset_sha256'],
            'snapshot_sha256':hashlib.sha256(raw).hexdigest(),'source_commit':snapshot['source_commit'],
            'index_revision':snapshot['index_revision'],'embedding_model':snapshot['embedding_model'],
            'embedding_digest':snapshot['embedding_digest'],'target_kind':snapshot.get('target_kind','entity'),'model':args.model,
            'settings':{'reasoning':args.reasoning,'max_output_tokens':args.max_output,'temperature':0},'cases':[],
            'method':'Identical raw test questions and up to 20 hybrid candidate chunks; compare first six chunks with app.retrieval.rag.rerank top six. Entity metrics deduplicate entity IDs within those chunks, so six chunks may represent fewer than six entities; passage metrics match exact passage IDs. Candidate pool is scored at 20 chunks. No routing, rewriting or answer generation. Unlisted sources are unjudged.',
            'hardware':'NVIDIA RTX A4000 16,376 MiB; shared host', 'metrics':{}}
    for case in snapshot['cases']:
        candidates=case['candidates']; targets=case.get('targets',case.get('positive_entity_ids')); kind=snapshot.get('target_kind','entity'); initial=score(candidates[:6],targets,kind)
        started=time.perf_counter(); error=None
        try:
            selected=rerank(case['question'],candidates,models,6)
            ranked=score(selected,targets,kind)
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'; ranked=None
        elapsed=round(1000*(time.perf_counter()-started),1)
        result['cases'].append({'id':case['id'],'question':case['question'],'targets':targets,
                                'candidate_ids':[c['id'] for c in candidates], 'candidate_pool':score(candidates,targets,kind), 'raw':initial,'reranked':ranked,
                                'rerank_ms':elapsed,'error':error})
        write(args.output,result)
        print(args.model,len(result['cases']),case['id'],'raw',initial['hit'],'reranked',None if ranked is None else ranked['hit'], 'ms',elapsed,'error',error,flush=True)
    for variant in ('candidate_pool','raw','reranked'):
        rows=[c[variant] for c in result['cases'] if c[variant] is not None]
        depth=20 if variant=='candidate_pool' else 6
        result['metrics'][variant]={'scored_cases':len(rows),'total_cases':len(result['cases']), 'chunk_depth':depth,
             f'hit_at_{depth}':round(statistics.mean(r['hit'] for r in rows),4),
             f'mrr_at_{depth}':round(statistics.mean(r['reciprocal_rank'] for r in rows),4),
             f'recall_at_{depth}':round(statistics.mean(r['recall'] for r in rows),4)} if rows else {'scored_cases':0}
    times=[c['rerank_ms'] for c in result['cases']]
    result['timing']={'first_case_ms':times[0], 'mean_after_first_ms':round(statistics.mean(times[1:]),1) if len(times)>1 else None}
    result['errors']=sum(bool(c['error']) for c in result['cases'])
    write(args.output,result); print(json.dumps({'metrics':result['metrics'],'timing':result['timing'],'errors':result['errors']}),flush=True)
if __name__=='__main__': main()
