#!/usr/bin/env python3
"""Evaluate a typed scope classifier as an experimental route, without changing the app."""
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
from app.ollama import ModelError,Ollama

CATEGORIES=('schema','records','runtime','unrelated','uncertain')
PROMPT=('Classify what the user is asking for. Only Microsoft CDM schema definitions are available. '
        'schema: asks about data-model entities, attributes, relationships or the abstract meaning of a banking concept. '
        'records: asks for actual stored customer, account, transaction or financial facts such as a balance or rate. '
        'runtime: asks whether or how a software application or workflow performs an operation. '
        'unrelated: greeting, personal story, or unrelated subject. uncertain: cannot decide. '
        'Examples: "What fields define a customer?" is schema. "List our customers" is records. '
        '"Does the application sync after saving?" is runtime. "Hello" is unrelated. '
        'Return exactly one category in JSON under kind.')


def classify(question,models,model):
    schema={'type':'object','properties':{'kind':{'type':'string','enum':list(CATEGORIES)}},
            'required':['kind'],'additionalProperties':False}
    value=models.chat_json(model,[{'role':'system','content':PROMPT},{'role':'user','content':question}],schema=schema,max_tokens=32)
    category=value.get('kind')
    if category not in CATEGORIES:raise ModelError('Invalid scope category')
    return category


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=a.baseline.read_bytes();baseline=json.loads(raw)
    models=Ollama(os.getenv('OLLAMA_BASE_URL','http://127.0.0.1:11434'),'qwen3-embedding:0.6b','qwen3:4b','qwen3:4b')
    policies={};metrics={}
    for model in ('qwen3:4b','qwen3:0.6b'):
        classify('What is a customer entity?',models,model);rows=[]
        for row in baseline['policies']['planner_4b']:
            t=time.perf_counter();kind='unrelated';error=None
            try:
                if row['probe_passed']:kind=classify(row['question'],models,model)
            except ModelError as exc:kind='error';error=str(exc)
            rows.append({**row,'kind':kind,'decision':'search' if kind=='schema' else ('fallback' if kind in ('uncertain','error') else 'no_search'),
                         'error':error,'latency_ms':round(1000*(time.perf_counter()-t),1)})
            print(f"{model} {len(rows)}/47 {row['case_id']} {kind}",flush=True)
        policies[model]=rows
        synth=[r for r in rows if r['dataset']=='synthetic_test'];m={}
        for group in ('in_scope','off_topic','unsupported_fact'):
            subset=[r for r in synth if r['category']==group];m[group+'_count']=len(subset)
            m[group+'_search_rate']=statistics.mean(r['decision']=='search' for r in subset)
        m['errors']=sum(r['kind']=='error' for r in rows);m['fallbacks']=sum(r['decision']=='fallback' for r in rows)
        m['warm_model_call_median_ms']=statistics.median(r['latency_ms'] for r in rows if r['probe_passed'])
        metrics[model]=m
    out={'baseline_routing_sha256':hashlib.sha256(raw).hexdigest(),
         'notes':'Classifier prompt was designed after the web failures; scores are diagnostic, not independent validation. No query rewrites, retrieval or answers were evaluated.',
         'prompt':PROMPT,'metrics':metrics,'policies':policies}
    a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
