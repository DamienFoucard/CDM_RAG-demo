# Web-question retrieval and routing pilot

Recorded on 2026-11-22 using the pinned Microsoft CDM snapshot and the local
Ollama models listed in the result files. The raw pilot results are saved in
the adjacent `evaluation/results/web-v1/` directory.

This is an exploratory convenience sample, **not an independent benchmark or
an estimate of production performance**. Its ten public-source scenarios
include source wording and paraphrases; five also have explicitly marked
schema-oriented adaptations. The 15 resulting inputs are not independent of
one another. Candidate entity/edge labels were mapped by the assistant after
prior experiments and have not been independently reviewed; they are examples,
not exhaustive relevance judgments. The original operational questions may
not be fully answerable from the extracted CDM schema. See
[the source list](web-question-candidates-v1.json) for wording, provenance,
and limits.

## Retrieval comparison

All five methods received the same raw questions, without routing or query
rewrites. Each was allowed up to four complete entity definitions plus their
direct extracted neighborhood, within the same 85,000-character context
budget. The two graph methods ranked up to 20 observed edges, then expanded
unique endpoints into complete entity contexts. The TransE model was trained
on all observed graph edges and used as a structural ranking signal; this is
not a link-prediction test. The graph variants are offline prototypes, not the
served search path. Timing is warm, local wall time and includes differing
work; the graph timing includes calculation of both graph rankings.

The two requests for actual account records or application behavior are
excluded from entity/edge recall: a related schema definition is not an answer
to those requests. Among the remaining 13 question forms, candidate entity
recall was averaged per question; eight forms have tentative candidate edges.

| Retrieval approach | Mean candidate-entity recall (13) | Mean candidate-edge recall (8) | Mean latency, ms (15) |
| --- | ---: | ---: | ---: |
| Hybrid passages, no reranker | 0.814 | 1.000 | 523 |
| Hybrid passages + 4B reranker | 0.763 | 1.000 | 3,807 |
| Neighborhood-enriched passages + reranker | 0.724 | 1.000 | 8,009 |
| Graph-assisted, reranker-seeded | 0.641 | 0.938 | 5,325 |
| Graph-assisted + TransE, reranker-seeded | 0.679 | 0.938 | 5,325 |

Direct neighborhoods can contain an edge even when its other endpoint is not
among the four expanded entities, so the edge figure is evidence coverage, not
a ranking or answer-accuracy score. These incomplete labels and the small,
partly duplicated sample do not support a general winner. In this pilot neither
reranking nor the tested graph methods improved candidate-entity recall over
raw hybrid retrieval.

## Lightweight routing

The same saved preflight scores and original planner prompt were used for 32
assistant-authored synthetic test questions and the 15 web forms. The test has
24 expected in-scope questions, four unrelated questions, and four requests
for facts not in the schema. Web routing is shown per case without treating
all partially answerable questions as binary ground truth. Model timings below
are medians of warm calls **excluding the shared embedding preflight**; they
are not end-to-end response times.

| Routing policy | In-scope searches / 24 | Unsupported searches / 4 | Warm median |
| --- | ---: | ---: | ---: |
| Existing 4B planner | 24 | 2 | 437 ms |
| 0.6B planner, same prompt | 24 | 4 | 108 ms |
| 0.6B classifier with raw-query search | 21 | 3 | 64 ms |

All three avoided search on the four unrelated synthetic questions because
those were stopped by the shared preflight. The 0.6B planner copied the literal
instruction placeholder `a short natural-language schema search` as its query
in 21 of 43 searches. That post-hoc count is not a complete query-quality
metric. A simulated uncertain/error fallback to the 4B planner was never
triggered by the classifier on these inputs and therefore did not rescue its
false negatives. The 4B planner remains the default.

A tighter planner prompt and a second, explicitly typed scope classifier were
tried **after observing the web failures**, so they are diagnostic rechecks,
not held-out validation. The tighter prompt rejected one previously accepted
synthetic in-scope example and still searched both unanswerable web cases. The
4B typed classifier rejected or deferred two of 24 synthetic in-scope examples;
it identified the runtime question but still treated the actual-records
question as schema. The 0.6B typed classifier rejected or deferred seven of
24 in-scope examples. Neither experimental policy is deployed.

An optional hosted Jev classifier is integrated, with local-planner fallback
for low confidence, invalid responses, or service errors. It sends question
text to its provider when explicitly enabled. It was tested with mocked
responses, not called live or benchmarked in this pilot; its performance cannot
be inferred from the local Qwen results.

## Out-of-scope answers and judge audit

With retrieval forced, all five answer-generation paths produced a definite
unsupported `no` for whether creating a D365 account automatically creates a
contact. None of the five joint-account answers clearly disclosed that the
schema contains no real account records. A live check of the existing demo
also routed both questions to search and returned the same kind of unsupported
D365 claim, despite citing an entity. A citation label alone does not verify
that a claim is supported.

The local 4B judge graded abbreviated, blinded evidence cards; those scores are
stored but **not validated** and the judge is the same model family used by the
reranker and generator. Its batch answer-grading calls returned invalid JSON.
A per-answer retry assigned the highest score to all ten answers, including
all five unsupported D365 claims. An assistant audit disagreed with all ten
scores; this audit is itself not independent expert review. The judge failed
this check and must not be used as an accuracy certification. Its recorded
scores and the audit are preserved together for inspection.

The failed scope experiments have not fixed the demo's out-of-scope weakness.
Do not use it to answer questions about actual records, operational workflows,
current rates, or product behavior without further safeguards and independently
reviewed evaluation.

## Reproduction

Install `requirements-graph-benchmark.txt` in addition to the base requirements
and provide the local models. The isolated data directory contains separate
copies of the baseline and neighborhood indices; it must not be a serving
volume. The retrieval script can create missing indexes and trains its TransE
model on the observed graph.

```sh
python scripts/benchmark_web.py --data-dir data/relationship-benchmark \
  --output evaluation/results/web-v1/retrieval.json
python scripts/benchmark_web_routing.py --data-dir data/relationship-benchmark \
  --output evaluation/results/web-v1/routing.json
python scripts/judge_web.py --data-dir data/relationship-benchmark \
  --input evaluation/results/web-v1/retrieval.json --output evaluation/results/web-v1/judge.json
python scripts/judge_web_answers.py --input evaluation/results/web-v1/retrieval.json \
  --output evaluation/results/web-v1/judge-answers.json
python scripts/audit_web_answers.py --result evaluation/results/web-v1/retrieval.json \
  --judge evaluation/results/web-v1/judge-answers.json --labels evaluation/web-answer-audit-v1.json \
  --output evaluation/results/web-v1/answer-audit.json
```

The exact original and tested recheck planner code snapshots are retained with
the result files. Do not interpret a re-run of the recheck script against
another planner as the same experiment.
