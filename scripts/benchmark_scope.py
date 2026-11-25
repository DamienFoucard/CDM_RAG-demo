#!/usr/bin/env python3
"""Benchmark the shared preflight and router, with or without the scope guard."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.retrieval.index import SchemaIndex
from app.runtime.ollama import ModelError,Ollama
from app.retrieval.rag import plan_routed_search


def metrics(rows):
    groups={name:[r for r in rows if r['expected']==name] for name in ('schema','records','runtime','mixed','unrelated')}
    m={'cases':len(rows),'errors':sum(r['error'] is not None for r in rows),
       'mean_total_ms':statistics.mean(r['total_ms'] for r in rows),
       'median_total_ms':statistics.median(r['total_ms'] for r in rows)}
    for name,cases in groups.items():
        if cases:
            m[name+'_count']=len(cases)
            m[name+'_searches']=sum(r['search'] is True for r in cases)
            m[name+'_no_search']=sum(r['search'] is False for r in cases)
            m[name+'_errors']=sum(r['search'] is None for r in cases)
    return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=ROOT/'evaluation/scope-questions-v1.json')
    p.add_argument('--split',choices=('all','development','test','regression'),required=True)
    p.add_argument('--policy',choices=('baseline','guarded'),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); raw=args.dataset.read_bytes(); data=json.loads(raw)
    cases=[r for r in data['cases'] if args.split=='all' or r['split']==args.split]
    models=Ollama(os.getenv('OLLAMA_BASE_URL','http://127.0.0.1:11434'),
                  'qwen3-embedding:0.6b','qwen3:4b','qwen3:4b')
    index=SchemaIndex(ROOT/'seed/catalog.json',args.data_dir/'baseline',models)
    if args.policy=='guarded':
        from app.retrieval.scope import assess_scope, SCOPE_PROMPT
    else:
        assess_scope=None
    # Warm up model paths; the warmup is not part of the cases or timing.
    index.semantic_probe('Which CDM entity describes a transaction?')
    plan_routed_search('Which CDM entity describes a transaction?',models)
    if assess_scope:assess_scope('Which CDM entity describes a transaction?',models)
    rows=[]
    for case in cases:
        t=time.perf_counter();error=None;decision=None;query='';scope=None;stage='preflight';similarity=None
        try:
            similarity=index.semantic_probe(case['question'])
            if similarity<0.58:decision=False
            else:
                if assess_scope:
                    stage='scope';scope=assess_scope(case['question'],models)
                    if scope!='schema':decision=False
                if decision is None:
                    stage='planner';decision,query,_=plan_routed_search(case['question'],models)
        except ModelError as exc:
            decision=None;error=str(exc)
        row={**case,'probe_similarity':round(similarity,6) if similarity is not None else None,
             'scope':scope,'search':decision,'query':query,'stage':stage,'error':error,
             'total_ms':round(1000*(time.perf_counter()-t),1)}
        rows.append(row); print(f"{args.policy} {len(rows)}/{len(cases)} {case['id']}: {decision} {scope}",flush=True)
    result={'dataset_id':data['dataset_id'],'dataset_status':data['status'],
            'dataset_sha256':hashlib.sha256(raw).hexdigest(),'split':args.split,'policy':args.policy,
            'source_commit':index.catalog['source_commit'],'embedding_digest':index.model_digest,
            'index_revision':index.index_revision,'planner_model':models.router_model,
            'threshold':0.58,'metrics':metrics(rows),'cases':rows,
            'note':'Route evaluation only; synthetic labels are assistant-authored and unreviewed; warm local wall time includes the preflight.'}
    if assess_scope:
        result.update({'scope_model':models.scope_model,
                       'scope_code_sha256':hashlib.sha256((ROOT/'app/retrieval/scope.py').read_bytes()).hexdigest(),
                       'scope_prompt_sha256':hashlib.sha256(SCOPE_PROMPT.encode()).hexdigest()})
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result['metrics'],indent=2),flush=True)

if __name__=='__main__':main()
