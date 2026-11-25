#!/usr/bin/env python3
"""Reconstruct cited source snippets for the previous ten audited answers."""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.retrieval.index import SchemaIndex
from app.runtime.ollama import Ollama
from app.retrieval.rag import expand_context


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    source=ROOT/'evaluation/results/web-v1/retrieval.json'; raw=source.read_bytes();result=json.loads(raw)
    models=Ollama(os.getenv('OLLAMA_BASE_URL','http://127.0.0.1:11434'),'qwen3-embedding:0.6b','qwen3:4b','qwen3:4b')
    index=SchemaIndex(ROOT/'seed/catalog.json',args.data_dir/'baseline',models)
    cases=[]
    for item in result['cases']:
        if not item['out_of_scope']:continue
        for variant,measurement in item['variants'].items():
            answer=measurement['answer']
            _,sources,_=expand_context(index,[{'entity_id':e} for e in measurement['entities']])
            labels=set(re.findall(r'\[(R\d+)\]',answer))
            available={source['citation']:source for source in sources}
            if not labels<=set(available):raise SystemExit('Could not reconstruct all cited labels')
            evidence=[f"[{source['citation']}] {source['text']}" for source in sources if source['citation'] in labels]
            cases.append({'id':item['case_id']+'-'+variant,'split':'regression','question':item['question'],
                          'answer':answer,'evidence':evidence,'expected_acceptable':False,
                          'kind':'records' if item['id']=='web-08' else 'runtime',
                          'note':'Cited source summaries only; full extracted contexts may contain more detail.'})
    live_path=ROOT/'evaluation/results/scope-v1/api-baseline.json'
    live=json.loads(live_path.read_bytes())
    for case in live['cases']:
        if case['id'] not in {'t07','t13','t21','t26'}:
            continue
        evidence=[]
        for source in case['sources']:
            entity=index.entity_context(source['entity_id'])
            parts=[f"{entity['entity_id']} ({entity['model']}): {entity['description']}"]
            attrs=[f"{a['name']}: {a.get('description','')}" for a in entity['attributes']]
            if attrs:parts.append('Attributes: '+'; '.join(attrs))
            edges=[f"{e['source_entity_id']} --{e['attribute']}--> {e['target_entity_id']}: {e.get('description','')}"
                   for e in entity['direct_neighbors']]
            if edges:parts.append('Extracted relationships: '+'; '.join(edges))
            evidence.append(f"[{source['entity_id']}] "+'\n'.join(parts))
        cases.append({'id':'api-baseline-'+case['id'],'split':'regression','question':case['question'],
                      'answer':case['answer'],'evidence':evidence,
                      'expected_acceptable':case['id']=='t26',
                      'kind':'citation_or_scope_failure',
                      'note':('Actual unmodified API answer with complete cited entity excerpts. Manually reviewed after the live paired run; the mixed-scope answer accurately describes the schema and states that specific records cannot be determined.' if case['id']=='t26' else
                              'Actual unmodified API answer with complete cited entity excerpts; manually reviewed after the live paired run.')})
    guard_path=ROOT/'evaluation/results/scope-v1/api-guard-final.json'; guard=json.loads(guard_path.read_bytes())
    for case in guard['cases']:
        if case['id']!='t07':continue
        evidence=[]
        for source in case['sources']:
            entity=index.entity_context(source['entity_id'])
            parts=[f"{entity['entity_id']} ({entity['model']}): {entity['description']}"]
            attrs=[f"{a['name']}: {a.get('description','')}" for a in entity['attributes']]
            if attrs:parts.append('Attributes: '+'; '.join(attrs))
            edges=[f"{e['source_entity_id']} --{e['attribute']}--> {e['target_entity_id']}: {e.get('description','')}"
                   for e in entity['direct_neighbors']]
            if edges:parts.append('Extracted relationships: '+'; '.join(edges))
            evidence.append(f"[{source['entity_id']}] "+'\n'.join(parts))
        cases.append({'id':'api-guard-t07','split':'regression','question':case['question'],
                      'answer':case['answer'],'evidence':evidence,'expected_acceptable':True,
                      'kind':'current_api_answer',
                      'note':'Actual final guarded API answer; checked against its full cited entity definition and relationship direction.'})
    out={'dataset_id':'judge-regression-v3','status':'assistant-audited real model outputs; not independently reviewed',
         'provenance':'Reconstructed citation summaries for ten previously audited forced-retrieval answers, four actual unmodified API answers, and one actual final guarded API relationship answer.',
         'source_result_sha256':hashlib.sha256(raw).hexdigest(),
         'baseline_api_result_sha256':hashlib.sha256(live_path.read_bytes()).hexdigest(),
         'guard_api_result_sha256':hashlib.sha256(guard_path.read_bytes()).hexdigest(),'cases':cases}
    args.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print('cases',len(cases))

if __name__=='__main__':main()
