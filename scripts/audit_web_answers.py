#!/usr/bin/env python3
"""Compare exploratory LLM answer grades to explicitly reviewed audit labels."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--judge',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=a.result.read_bytes();result=json.loads(raw);judge=json.loads(a.judge.read_bytes());labels=json.loads(a.labels.read_bytes())
    if judge['source_result_sha256']!=hashlib.sha256(raw).hexdigest():raise SystemExit('Mismatched retrieval source')
    possible={(c['case_id'],v) for c in result['cases'] if c['out_of_scope'] for v in c['variants']}
    predicted={(r['case_id'],r['variant']):r for r in judge['cases']}
    expected={(r['case_id'],r['variant']):r for r in labels['cases']}
    if set(expected)!=possible or set(predicted)!=possible:raise SystemExit('Audit does not cover every unanswerable answer')
    rows=[]
    for key in sorted(possible):
        observed=predicted[key]['grade']; reviewed=expected[key]['grade']
        rows.append({'case_id':key[0],'variant':key[1],'judge_grade':observed,'reviewed_grade':reviewed,
                     'disagreement':observed is not None and observed!=reviewed})
    variants=sorted({r['variant'] for r in rows}); metrics={}
    for variant in variants:
        items=[r for r in rows if r['variant']==variant];graded=[r for r in items if r['judge_grade'] is not None]
        metrics[variant]={'reviewed_answer_cases':len(items),'reviewed_mean_grade':statistics.mean(r['reviewed_grade'] for r in items),
                          'judge_grade_disagreements':sum(r['disagreement'] for r in graded), 'judge_graded_cases':len(graded)}
    out={'status':labels['status'],'rubric':labels['rubric'],'notes':labels['notes'],
         'source_result_sha256':hashlib.sha256(raw).hexdigest(),
         'judge_result_sha256':hashlib.sha256(a.judge.read_bytes()).hexdigest(),
         'audit_labels_sha256':hashlib.sha256(a.labels.read_bytes()).hexdigest(), 'metrics':metrics,'cases':rows}
    a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
