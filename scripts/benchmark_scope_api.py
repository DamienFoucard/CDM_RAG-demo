#!/usr/bin/env python3
"""Exercise real FastAPI requests on a paired subset of the scope questions."""
import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

IDS=('r01','r02','t01','t07','t13','t21','t26','t29')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-url',required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    raw=args.dataset.read_bytes();dataset=json.loads(raw);by_id={c['id']:c for c in dataset['cases']}
    with urllib.request.urlopen(args.base_url+'/api/v1/status',timeout=15) as response:status=json.load(response)
    rows=[]
    for case_id in IDS:
        case=by_id[case_id];request=urllib.request.Request(args.base_url+'/api/v1/query',
            data=json.dumps({'question':case['question']}).encode(),headers={'Content-Type':'application/json'},method='POST')
        start=time.perf_counter();error=None
        try:
            with urllib.request.urlopen(request,timeout=180) as response:answer=json.load(response)
        except (urllib.error.URLError,TimeoutError,ValueError) as exc:
            answer={};error=type(exc).__name__
        rows.append({**case,'route':answer.get('route'),'answer':answer.get('answer'),
                     'sources':[{'entity_id':s.get('entity_id'),'uri':s.get('uri')} for s in answer.get('sources',[])],
                     'retrieval':answer.get('retrieval',{}),'error':error,'latency_ms':round(1000*(time.perf_counter()-start),1)})
        print(f"{args.base_url} {len(rows)}/{len(IDS)} {case_id}: {rows[-1]['route']} {error or ''}",flush=True)
    schema=[r for r in rows if r['expected']=='schema'];negative=[r for r in rows if r['expected']!='schema']
    result={'dataset_id':dataset['dataset_id'],'dataset_sha256':hashlib.sha256(raw).hexdigest(),
            'status':status,'note':'Real API behavior on eight selected assistant-labeled cases; no automatic semantic answer grading.',
            'metrics':{'cases':len(rows),'errors':sum(bool(r['error']) for r in rows),
                       'schema_cases':len(schema),'schema_cited_answers':sum(r['route']=='cdm' and bool(r['sources']) for r in schema),
                       'negative_cases':len(negative),'negative_declined_no_sources':sum(r['route']=='no_search' and not r['sources'] for r in negative),
                       'median_ms':statistics.median(r['latency_ms'] for r in rows)},'cases':rows}
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result['metrics'],indent=2))

if __name__=='__main__':main()
