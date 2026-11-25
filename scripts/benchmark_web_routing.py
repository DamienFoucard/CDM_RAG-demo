#!/usr/bin/env python3
"""Compare local planner sizes and a light classify/direct-search policy on the same probes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.retrieval.index import SchemaIndex
from app.runtime.ollama import ModelError, Ollama
from app.retrieval.rag import plan_search
from scripts.benchmark_web import web_cases


def classify_light(question, models):
    schema = {'type':'object', 'properties':{'decision':{'type':'string','enum':['search','no_search','uncertain']}},
              'required':['decision'],'additionalProperties':False}
    result = models.chat_json('qwen3:0.6b', [
        {'role':'system','content':(
            'Classify whether searching Microsoft CDM banking and common schema definitions could help answer the message. '
            'Choose search for entities, attributes, relationships, or banking data models. '
            'Choose no_search for greetings, unrelated personal stories, and requests for actual personal account records. '
            'Choose uncertain if unsure. Schema definitions cannot establish real customer records or software runtime behavior. '
            'Return only JSON with decision: search, no_search, or uncertain.')},
        {'role':'user','content':question}],schema=schema,max_tokens=30)
    decision = result.get('decision')
    if decision not in ('search','no_search','uncertain'):
        raise ModelError('The lightweight classifier returned an invalid decision.')
    return decision


def mean(values):
    return statistics.mean(values) if values else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--web',type=Path,default=ROOT/'evaluation/web-question-candidates-v1.json')
    p.add_argument('--synthetic',type=Path,default=ROOT/'evaluation/results/v2/demo-qwen-test.json')
    args=p.parse_args()
    raw=args.web.read_bytes(); web=web_cases(json.loads(raw)); synthetic=json.loads(args.synthetic.read_text())
    models=Ollama(os.getenv('OLLAMA_BASE_URL','http://127.0.0.1:11434'),
                  'qwen3-embedding:0.6b','qwen3:4b','qwen3:4b')
    with httpx.Client(timeout=10,trust_env=False) as client:
        response=client.get(models.base_url+'/api/tags'); response.raise_for_status()
        model_digests={m['name']:m['digest'] for m in response.json()['models']
                       if m['name'] in ('qwen3:4b','qwen3:0.6b','qwen3-embedding:0.6b')}
    if len(model_digests)!=3:
        raise SystemExit('Both router models and the embedding model must be installed')
    index=SchemaIndex(ROOT/'seed/catalog.json',args.data_dir/'baseline',models)
    cases=[]
    for c in synthetic['cases']+synthetic['out_of_scope']:
        cases.append({'case_id':c['id'],'dataset':'synthetic_test','question':c['question'],
                      'category':'in_scope' if 'positive_entity_ids' in c else c['category'],
                      'expected_search':'positive_entity_ids' in c,'probe_similarity':c['probe_similarity']})
    for c in web:
        score=index.semantic_probe(c['question'])
        cases.append({'case_id':c['case_id'],'dataset':'web','question':c['question'],
                      'category':'data_or_runtime_unanswerable' if c['out_of_scope'] else 'schema_or_partial',
                      'expected_search':None,'probe_similarity':score,
                      'candidate_entities':c.get('candidate_entities',[])})
    threshold=synthetic['gate']['threshold']; outputs={}
    for policy in ('planner_4b','planner_0_6b','classify_0_6b'):
        models.router_model='qwen3:4b' if policy=='planner_4b' else 'qwen3:0.6b'
        # Discard warm-up timing, because loading can dominate a single call.
        if policy.startswith('planner'): plan_search('What is an account?',models)
        else: classify_light('What is an account?',models)
        rows=[]
        for case in cases:
            record={**case,'probe_passed':case['probe_similarity']>=threshold}
            t=time.perf_counter(); decision='no_search'; query=''; error=None
            if record['probe_passed']:
                try:
                    if policy=='classify_0_6b':
                        decision=classify_light(case['question'],models)
                        query=case['question'] if decision=='search' else ''
                    else:
                        search,query=plan_search(case['question'],models)
                        decision='search' if search else 'no_search'
                except ModelError as exc:
                    decision='error';error=str(exc)
            record.update(decision=decision,query=query,error=error,
                          latency_ms=round(1000*(time.perf_counter()-t),1))
            rows.append(record)
            print(f"{policy} {len(rows)}/{len(cases)} {case['case_id']} {decision}",flush=True)
            outputs[policy]=rows
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps({'partial':True,'policies':outputs},indent=2,ensure_ascii=False)+'\n')
    # Cascade uses separately measured planner times/decisions and falls back on
    # uncertain OR malformed classifier results. It does not pretend to time a real chained call.
    cascade=[]
    for light, large in zip(outputs['classify_0_6b'],outputs['planner_4b']):
        use_fallback=light['decision'] in ('uncertain','error')
        row={**light,'decision':large['decision'] if use_fallback else light['decision'],
             'query':large['query'] if use_fallback else light['query'],
             'fallback':use_fallback,'latency_ms':round(light['latency_ms']+(large['latency_ms'] if use_fallback else 0),1)}
        cascade.append(row)
    outputs['classify_0_6b_with_4b_fallback_estimate']=cascade
    metrics={}
    for policy,rows in outputs.items():
        synth=[r for r in rows if r['dataset']=='synthetic_test']
        groups={g:[r for r in synth if r['category']==g] for g in ('in_scope','off_topic','unsupported_fact')}
        met={'cases':len(rows),'errors':sum(r['decision']=='error' for r in rows),
             'uncertain':sum(r['decision']=='uncertain' for r in rows),
             'fallbacks':sum(r.get('fallback',False) for r in rows),
             # This known prompt-placeholder failure was noticed while inspecting
             # saved outputs. It is not an exhaustive query-quality measure.
             'literal_placeholder_query_count':sum(r['decision']=='search' and
                 r['query'].strip().lower()=='a short natural-language schema search' for r in rows),
             'warm_model_call_median_ms':statistics.median([r['latency_ms'] for r in rows if r['probe_passed']]),
             'mean_policy_latency_ms_excluding_probe':mean([r['latency_ms'] for r in rows])}
        for name,items in groups.items():
            met[name+'_count']=len(items)
            met[name+'_search_rate']=mean([r['decision']=='search' for r in items])
        metrics[policy]=met
    result={'dataset_id':'web-and-synthetic-routing-v1','web_dataset_sha256':hashlib.sha256(raw).hexdigest(),
            'synthetic_dataset_sha256':synthetic['dataset_sha256'],'probe_threshold':threshold,
            'model_digests':model_digests,
            'notes':'Synthetic test labels are unreviewed; web expected route is intentionally unscored. '
                    'No-search is a product policy, not proof that schema evidence is useless. '
                    'Timing is local warm-model wall time excluding the shared probe. Cascade is calculated from separately timed calls.',
            'metrics':metrics,'policies':outputs}
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
