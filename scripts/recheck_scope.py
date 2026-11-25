#!/usr/bin/env python3
"""Repeat routing after a scope-prompt change against the saved, shared preflight scores."""
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
from app.runtime.ollama import ModelError, Ollama
from app.retrieval.rag import plan_search


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=a.baseline.read_bytes();baseline=json.loads(raw)
    models=Ollama(os.getenv('OLLAMA_BASE_URL','http://127.0.0.1:11434'), 'qwen3-embedding:0.6b','qwen3:4b','qwen3:4b')
    plan_search('What is a CDM account?',models)
    rows=[]
    for c in baseline['policies']['planner_4b']:
        t=time.perf_counter();error=None
        try:
            decision,query=plan_search(c['question'],models) if c['probe_passed'] else (False,'')
        except ModelError as exc:
            decision,query,error=None,'',str(exc)
        rows.append({**c,'previous_decision':c['decision'],
                     'decision':'error' if decision is None else ('search' if decision else 'no_search'),
                     'query':query,'error':error,'latency_ms':round(1000*(time.perf_counter()-t),1)})
        print(f"{len(rows)}/{len(baseline['policies']['planner_4b'])} {c['case_id']} {rows[-1]['decision']}",flush=True)
    synth=[r for r in rows if r['dataset']=='synthetic_test']; metrics={}
    for group in ('in_scope','off_topic','unsupported_fact'):
        subset=[r for r in synth if r['category']==group]
        metrics[group+'_count']=len(subset)
        metrics[group+'_search_rate']=statistics.mean(r['decision']=='search' for r in subset)
    metrics['errors']=sum(r['decision']=='error' for r in rows)
    metrics['warm_model_call_median_ms']=statistics.median(r['latency_ms'] for r in rows if r['probe_passed'])
    out={'baseline_routing_sha256':hashlib.sha256(raw).hexdigest(),
         'rag_code_sha256':hashlib.sha256((ROOT/'app/retrieval/rag.py').read_bytes()).hexdigest(),
         'notes':'Prompt changed after observing web failures, so this is a diagnostic recheck, not held-out validation.',
         'metrics':metrics,'cases':rows}
    a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
