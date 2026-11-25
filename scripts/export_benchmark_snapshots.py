#!/usr/bin/env python3
"""Regenerate the saved benchmark drill-down snapshots from local run artifacts.

The inputs are local source-derived experiment records, not model calls. The
exporter checks dataset membership, questions, evidence hashes, index identity,
and rankings, and deliberately exports only the fields used by the UI.

Example: .venv/bin/python scripts/export_benchmark_snapshots.py
Use --run-dir and --output-dir to select other local artifacts and destinations.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from app.schema.corpus import load_catalog, render_chunks
from app.schema.graph import SchemaGraph

run = root / 'data/macstudio-runs'
out = root / 'evaluation/results' 
source_keys=('citation','id','passage_id','source_id','source_type','source_label','title','text','uri','kind','entity_id')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text())
def clean_source(s): return {k:s[k] for k in source_keys if k in s and isinstance(s[k],str)}
def canon(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def parse_sse(path,kind):
 events=[json.loads(line[6:]) for line in path.read_text().splitlines() if line.startswith('data: ')]
 out=[e['result'] for e in events if e.get('type')==kind];assert len(out)==1
 return out[0]

def dataset(name):
 p=root/'evaluation'/name; d=load(p);return d,{q['id']:q for q in d['questions']}
def write(path,obj):
 temp=path.with_name(path.name+'.tmp')
 temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
 temp.replace(path)

def main():
 global run, out
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--run-dir', type=Path, default=run, help='Directory containing local saved run artifacts')
 parser.add_argument('--output-dir', type=Path, default=out, help='Directory for sanitized JSON snapshots')
 args=parser.parse_args(); run=args.run_dir; out=args.output_dir
 out.mkdir(parents=True, exist_ok=True)
 base,bq=dataset('answer-benchmark-v1.json');paired,pq=dataset('paired-answer-challenge-candidate-v1.json')
 original_path=run/'answer-events.sse';original=parse_sse(original_path,'benchmark_completed');assert original['dataset_id']==base['dataset_id'] and original['source_commit']==base['source_commit']
 rows=[];provenance=[]
 def track(p,label,model_info=None):
  d={'artifact':p.name,'sha256':sha(p),'description':label}
  if model_info:d['models']=model_info
  provenance.append(d)
 track(original_path,'Original answer benchmark, saved complete run')
 for r in original['rows']:
  q=bq[r['question_id']]; assert r['question']==q['question'] and r['reference_answer']==q['reference_answer']
  src=[clean_source(s) for s in r.get('retrieved_sources',[])]
  rows.append({'dataset_id':'answer-benchmark-v1','split':q['split'],'model':r['model'],'case_id':q['id'],'arm':'raw','question':q['question'],'reference_answer':q['reference_answer'],'query':q['question'],'status':'error' if r.get('error') else ('fallback' if r.get('generator')=='extractive-fallback' else 'generated'),'answer':r.get('answer') or '', 'error':r.get('error') or '', 'retrieved_sources':src,'retrieval_candidate_ids':r.get('retrieval',{}).get('candidate_entity_ids',[]),'retrieval_note':'Saved candidate list identifies entities; full passage text shown for saved sources supplied to the generator.','run_id':r['run_id']})

 paired_rows={}; paired_case_evidence={}
 for filename in ('paired-answer-revised-dev.json','paired-answer-revised-test.json'):
  p=run/filename;d=load(p);assert d['dataset_id']==paired['dataset_id'] and d['source_commit']==paired['source_commit'];track(p,'Revised paired retrieval and answer run',{k:v.get('digest') for k,v in d['models'].items() if isinstance(v,dict) and v.get('digest')})
  for r in d['rows']:
   q=pq[r['id']];assert r['question']==q['question'] and r['reference_answer']==q['reference_answer']
   for arm,a in r['arms'].items():
    assert arm in ('raw','rewrite');key=(r['id'],arm);assert key not in paired_rows;paired_rows[key]=a
    evidence_hash=canon({'contexts':a['contexts'],'sources':a['sources']});paired_case_evidence[key]=evidence_hash
    rows.append({'dataset_id':'paired-rag-revised','split':q['split'],'model':d['models']['answer']['name'],'case_id':q['id'],'arm':arm,'question':q['question'],'reference_answer':q['reference_answer'],'query':a['query'],'status':a['status'],'answer':a.get('answer') or '', 'error':a.get('error') or '', 'retrieved_sources':[clean_source(s) for s in a['sources']], 'retrieval_candidates':[clean_source(s) for s in a.get('candidates',[])], 'retrieval_note':'Candidates are the saved ranked retrieval results; source passages are the subset supplied to the generator.', 'evidence_sha256':evidence_hash})

 for filename in ('fixed-evidence-generator-comparison.json','fixed-evidence-generator-14b.json','fixed-evidence-gpt-oss-20b-low-2048.json','answer-lfm24b-12.json','answer-qwen35-27b-thinking.json'):
  p=run/filename;d=load(p);model_digests={name:meta['digest'] for name,meta in d['models'].items() if meta.get('digest')};track(p,'Fixed saved evidence answer generation',model_digests)
  cases={(c['case_id'],c['arm']):c for c in d['cases']}; assert len(cases)==len(d['cases'])
  for key,c in cases.items():
   q=pq[key[0]];assert c['question']==q['question'] and c['reference_answer']==q['reference_answer']
   assert c['evidence_sha256']==canon({'contexts':c['contexts'],'sources':c['sources']})==paired_case_evidence[key]
  seen=set()
  for r in d['rows']:
   key=(r['case_id'],r['arm']);model=r['model'];assert (key,model) not in seen;seen.add((key,model));c=cases[key];assert r['evidence_sha256']==c['evidence_sha256'];q=pq[key[0]]
   rows.append({'dataset_id':'fixed-evidence','split':q['split'],'model':model,'case_id':q['id'],'arm':key[1],'question':q['question'],'reference_answer':q['reference_answer'],'query':c.get('query') or '', 'status':r['status'],'answer':r.get('answer') or '', 'error':r.get('error') or '', 'retrieved_sources':[clean_source(s) for s in c['sources']], 'retrieval_note':'Frozen passages supplied to each answer model; this experiment did not rerun retrieval.', 'evidence_sha256':c['evidence_sha256']})
  # Interrupted 27B attempts: expose selected but not-run rows honestly.
  for model in d['models']:
   for key,c in cases.items():
    if (key,model) not in seen:
     q=pq[key[0]];rows.append({'dataset_id':'fixed-evidence','split':q['split'],'model':model,'case_id':q['id'],'arm':key[1],'question':q['question'],'reference_answer':q['reference_answer'],'query':c.get('query') or '', 'status':'not_run','answer':'','error':'Not completed; experiment was interrupted.','retrieved_sources':[clean_source(s) for s in c['sources']], 'retrieval_note':'Frozen passages selected for the attempted experiment; this arm was not run.','evidence_sha256':c['evidence_sha256']})

 assert len(rows)==18+12+60+8,len(rows)
 assert len({(r['dataset_id'],r['model'],r['case_id'],r['arm']) for r in rows})==len(rows)
 answer={'schema_version':1,'description':'Public-source-derived benchmark excerpts and saved answers only. Original and fixed-evidence experiments use distinct datasets.','catalog_revision':base['source_commit'],'datasets':{'answer-benchmark-v1':{'dataset_id':base['dataset_id'],'dataset_sha256':sha(root/'evaluation/answer-benchmark-v1.json')},'paired-rag-revised':{'dataset_id':paired['dataset_id'],'dataset_sha256':sha(root/'evaluation/paired-answer-challenge-candidate-v1.json')},'fixed-evidence':{'dataset_id':paired['dataset_id'],'dataset_sha256':sha(root/'evaluation/paired-answer-challenge-candidate-v1.json')}},'provenance':provenance,'rows':rows}
 write(out/'generation-output-snapshot.json',answer)

 challenge,cq=dataset('reformulation-challenge-candidate-v1.json');catalog=load_catalog(root/'seed/catalog.json');assert catalog['source_commit']==challenge['source_commit'];SchemaGraph(catalog['entities'], catalog['source_commit'], semantics=catalog.get('semantics', {}));chunks=render_chunks(catalog);byid={c['id']:c for c in chunks};assert len(byid)==len(chunks)
 # Recompute exactly the same chunk/index identity used in SchemaIndex before using text.
 nomic=load(run/'retrieval-qwen35-27b-thinking-test-rest.json')['embedding']; index_payload={'chunks':chunks,'model':nomic['name'],'model_digest':nomic['digest'],'format':'cdm-passages-v3','embedding_fragment_chars':3000,'embedding_format':'nomic-search-v1'}; index_hash=hashlib.sha256(json.dumps(index_payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest(); assert index_hash==load(run/'retrieval-qwen35-27b-thinking-test-rest.json')['index_identity'],index_hash
 def passage(pid):
  assert pid in byid, f'Unresolved passage {pid}'
  return clean_source(byid[pid])
 inputs=[];retrows=[]
 for filename in ('retrieval-qwen35-27b-thinking-test.json','retrieval-qwen35-27b-thinking-test-rest.json'):
  p=run/filename;d=load(p);assert d['source_commit']==challenge['source_commit'] and d['split']=='test';inputs.append({'artifact':p.name,'sha256':sha(p),'model':d['model']['name'],'model_digest':d['model']['digest'],'index_identity':d['index_identity'],'status':d['status']})
  for r in d['rows']:
   q=cq[r['question_id']];assert q['split']=='test' and r['question']==q['question'] and r['status']=='routed'
   ids=r['fused_ids'];assert ids[:6]==r['adaptive']['ranked_passage_ids'];assert all(x in byid for x in ids)
   matched=[x for x in ids[:6] if x in q['positive_passage_ids']];assert set(matched)==set(r['adaptive']['matched_passage_ids'])
   retrows.append({'strategy':'adaptive-qwen3.5-27b','model':d['model']['name'],'split':'test','question_id':q['id'],'question':q['question'],'initial_query':q['question'],'search_again':r['decision']['search_again'],'followup_query':r['decision']['query'],'status':r['status'],'ranked_passages':[passage(i) for i in ids],'ranked_passage_ids':ids,'fused_ids':ids,'matched_passage_ids':matched,'target_ids':q['positive_passage_ids'],'first_pass_ids':r['first_pass_ids'],'followup_ids':r['followup_ids']})
 assert len(retrows)==6 and len({r['question_id'] for r in retrows})==6
 for filename, strategy in (('graph-expansion-9b.json','graph-qwen3.5-9b'),('graph-expansion-12b.json','graph-gemma4-12b')):
  p=run/filename;d=load(p);assert d['source_commit']==challenge['source_commit'];inputs.append({'artifact':p.name,'sha256':sha(p),'model':d['query_model'],'status':'offline replay'})
  test=[r for r in d['rows'] if r['split']=='test'];assert len(test)==6
  for r in test:
   q=cq[r['question_id']];assert q['question']==r['question'] and q['positive_passage_ids']==r['gold_ids'];ids=r['expanded_ids'];assert all(x in byid for x in ids)
   matched=[x for x in ids[:6] if x in q['positive_passage_ids']];assert abs(len(matched)/len(q['positive_passage_ids'])-r['graph_expansion']['recall_at_6'])<1e-10
   retrows.append({'strategy':strategy,'model':d['query_model'],'split':'test','question_id':q['id'],'question':q['question'],'initial_query':q['question'],'search_again':None,'followup_query':'','status':'offline replay','ranked_passages':[passage(i) for i in ids],'ranked_passage_ids':ids,'fused_ids':ids,'matched_passage_ids':matched,'target_ids':q['positive_passage_ids'],'first_pass_ids':r['first_pass_ids'],'followup_ids':[]})
 assert len(retrows)==18 and len({(r['strategy'],r['question_id']) for r in retrows})==18
 retrieval={'schema_version':1,'description':'Saved ranked retrieval output and offline graph-expansion replay. Passage texts reconstructed from the exact pinned catalog; 27B run was split by memory-guard interruption.','catalog_revision':catalog['source_commit'],'dataset_id':challenge['dataset_id'],'dataset_sha256':sha(root/'evaluation/reformulation-challenge-candidate-v1.json'),'catalog_sha256':sha(root/'seed/catalog.json'),'provenance':inputs,'rows':retrows}
 write(out/'retrieval-qwen35-27b-output-snapshot.json',retrieval)
 print('generation rows',len(rows),'bytes',(out/'generation-output-snapshot.json').stat().st_size)
 print('retrieval rows',len(retrows),'bytes',(out/'retrieval-qwen35-27b-output-snapshot.json').stat().st_size)

if __name__ == '__main__':
 main()
