# Scope guard and answer judge pilot

## Scope guard

The index contains Microsoft CDM definitions, not customer, account, transaction,
or current bank records and not deployed application state. A lightweight local
scope check now runs after the semantic preflight and before query planning. It
allows schema questions and abstract banking concept definitions/comparisons;
it declines records, application behavior, mixed requests, unrelated requests,
unclear intent, and classifier errors. Declined requests return no retrieved
sources. `SCOPE_MODEL` defaults to `qwen3:4b`.

The paired route comparison uses the same questions, semantic gate, and planner
before and after the guard. It measures routing, not answer quality.

| Dataset | Policy | Correct route | Schema searched | Non-schema searches |
| --- | --- | ---: | ---: | ---: |
| Synthetic held-out (32) | Planner only | 20/32 (62.5%) | 12/12 | 12/20 (60%) |
| Synthetic held-out (32) | Scope guard | 32/32 (100%) | 12/12 | 0/20 |
| Public-question pilot (15) | Planner only | 9/15 (60%) | 9/9 | 6/6 (100%) |
| Public-question pilot (15) | Scope guard | 15/15 (100%) | 9/9 | 0/6 |

The public-question set contains source-derived wording and adapted schema
questions. Its policy labels were written after reviewing those questions; they
are not independent relevance judgments. The 32-question synthetic test labels
are also assistant-authored and unreviewed. These scores are pilot evidence,
not a claim of general accuracy.

In an eight-request API comparison, both policies returned cited results for
the two schema questions. The old API declined only one of six records/runtime/
mixed requests without sources; the guarded API declined all six without
sources. The current guarded API correctly says that `Contact.accountId`
identifies the associated account. The older answer reversed that direction; the
scope guard itself does not guarantee factual correctness for in-scope answers.

The held-out routing run had 605 ms mean and 477 ms median warm local routing
time. This excludes answer generation and is not end-to-end latency. Scope
routing errors were zero in the synthetic and public-question runs. The guard
fails closed if its classifier errors.

## Local classifier and query-planner timing comparison

The production scope classifier and local query planner were compared using
the same model, with the planner called only when scope classification returned
`schema`. The run used the 22-question development split, warmed each model
after measuring one cold-start classification, and recorded both wall time and
Ollama's native load, prefill, and decode durations.

| Model | Weight file | Cold scope request | Cold model load | Warm scope mean / p95 | Planner mean / p95 | Full route mean / p95 | Scope categories | Search decisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3.5:4b` | 3.70 GiB | 9.97 s | 4.92 s | 2.45 / 3.07 s | 1.46 / 1.85 s | 2.98 / 4.47 s | 22/22 (100%) | 22/22 (100%) |
| `qwen3:0.6b` | 0.49 GiB | 2.00 s | 1.32 s | 0.35 / 0.39 s | 0.25 / 0.37 s | 0.44 / 0.68 s | 16/22 (72.7%) | 22/22 (100%) |

The 0.6B model classified all eight schema questions as searchable and the
query planner chose search for all eight. Its category errors were three of
four runtime questions and all three unrelated questions, which it labeled
`mixed`. That preserves the no-search decision on this split, but the app gives
different explanations for those categories, so the smaller model is not yet a
safe replacement for the full scope behavior. It rewrote six of eight schema
queries; query quality was not graded. These assistant-authored development
labels are not an independent accuracy estimate, and the test split remains
reserved.

The [per-question JSON run](results/scope-v1/scope-classifier-development.json)
contains the model digests, exact questions and predictions, prompt/data/code
hashes, cold-start timings, stage wall times, and Ollama token and duration
counters. `scripts/benchmark_scope_classifier.py` reproduces it.

## Answer-grounding judge

The updated local judge uses `qwen3.5:9b-q4_K_M` and checks cited evidence,
relationship direction, coverage of the full question, and explicit limits on
what a schema-only collection can establish. It sees only excerpts cited in the
answer; an uncited excerpt cannot validate a claim. Unknown citation labels are
rejected in code, and an internally contradictory approval becomes uncertain.
Trusted collection-scope metadata can support a clear refusal, but cannot
support CDM schema claims.

| Set | Cases | Correct | Errors/uncertain | Median judge time |
| --- | ---: | ---: | ---: | ---: |
| Development | 12 | 11/12 | 0 | 8.30 s |
| Held-out balanced test | 20 | 20/20 | 0 | 4.88 s |
| Real-output regression | 15 | 15/15 | 0 | 12.65 s |

The development disagreement is a short answer that declines to list customer
records but only briefly describes `Contact`; the judge considers that schema
description incomplete. The answer's adequacy is debatable, which illustrates
why the labels need independent review. The regression set includes 13
unacceptable and two acceptable outputs, including paired old/new relationship
answers; it is a challenge set, not a balanced accuracy estimate.

On 16 held-out cases whose question, answer, evidence, and label were unchanged,
the previous 9B rubric accepted one answer with a wrong citation (15/16 correct).
The new judge filtered out uncited evidence and rejected that answer (16/16).
The final held-out set adds two cases and corrects two ambiguous examples; it
scores 20/20. The previous 4B judge scored 10/12 on development and falsely
accepted two unsupported answers. These small, assistant-reviewed sets do not
establish production-level judge reliability.

## Reproduction and tracking

Run the route comparison and judge using the saved inputs and scripts:

```sh
python scripts/compare_scope_routes.py \
  --results-dir evaluation/results \
  --web-results-dir evaluation/results/web-v1 \
  --output evaluation/results/scope-v1/route-comparison.json

python scripts/benchmark_scope.py \
  --data-dir data --split test --policy guarded \
  --output evaluation/results/scope-v1/guard-test.json

python scripts/judge_grounding_v3.py \
  --dataset evaluation/judge-questions-v2.json \
  --split test --model qwen3.5:9b-q4_K_M \
  --output evaluation/results/scope-v1/judge-v3-9b-test.json
```

The exact datasets, per-question decisions, answers, sources, model digests,
code/prompt hashes, and comparisons are saved as JSON files alongside this
report.

## Saved results

- [Paired route scorecard](results/scope-v1/route-comparison.json)
- [Guarded held-out routing](results/scope-v1/guard-test.json)
- [Public-question routing](results/scope-v1/web-guard.json)
- [Local classifier and planner timing comparison](results/scope-v1/scope-classifier-development.json)
- [Paired live API answers](results/scope-v1/api-guard-final.json)
- [Judge development run](results/scope-v1/judge-v3-9b-development.json)
- [Judge held-out run](results/scope-v1/judge-v3-9b-test.json)
- [Judge real-output regression](results/scope-v1/judge-v3-9b-regression.json)
