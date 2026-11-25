# MacStudio handoff: evaluate evidence-informed retrieval

## Objective

Find out whether a larger local model can use the first retrieved schema passages to formulate a useful follow-up search. Compare retrieval quality and cost with the same raw first search, then inspect the generated query and individual passages. The current MacBook results do not establish that model size is the cause of poor rewrites: query formulation, passage truncation, fusion, retrieval, and incomplete labels can all affect the outcome.

This document records the original MacBook-to-Mac Studio handoff. The retrieval follow-up router described below remains an **evaluation experiment**. Since the handoff, the app also gained a separate optional evidence-aware decision router that can inspect an initial search before deciding; the local Mac Studio uses it, while the repository default remains the earlier local planner. An optional judge can request another search after answer generation but is disabled by default. See the current [RAG](rag/README.md) and [benchmark](benchmarking/README.md) guides.

## Evidence to reproduce

The [v2 dataset](../evaluation/adaptive-retrieval-v2.json) contains 31 cases: 16 earlier cases and 15 additional cases (seven development and eight test). The [MacBook v2 report](../evaluation/results/adaptive-retrieval-v2/README.md) and its JSON records cover those 15 additions. Labels are source-checked but incomplete, largely synthetic, and not independently reviewed; the test set is published and not blind. The [v1 report](../evaluation/results/adaptive-retrieval-v1/README.md) used a different router excerpt limit and fusion cutoff, so do not pool its scores with v2.

The user's mortgage question is `adaptive-v2-dev-mortgage-holding`. The raw query returned one of six labeled evidence passages in its top six, while the user's independently written query returned three; at 20 passages the counts were four and three respectively. The human query names relevant schema objects and is **not** a model output or an assumed optimum. It is stored on that case as `human_reference_query` and is evaluated separately; the router is not given it or the labels. The indexed schema includes the `financialHoldingRole` description and role values, customer and holding links, and account/holding definitions. It contains no customer records or complete mortgage application entity.

On the eight new test cases, the MacBook's Gemma 3 4B router produced the same average recall@6 before and after adaptive fusion (0.417), and recall@20 fell from 0.792 to 0.688. Mean elapsed stage time was 0.48 s for the first search and 17.57 s for adaptive retrieval on an Apple M1 with 8 GiB unified memory. The embedder was unloaded after each use and the router after each decision. These are retrieval scores and machine-specific timings, not answer-quality scores.

## Prepare the MacStudio

1. If using shared hardware, arrange an exclusive model-work window and preserve active jobs before loading models. The CDM RAG runner calls Ollama directly and does not acquire an external scheduler lease.
2. Clone or update this repository and record `git rev-parse HEAD`. The catalog at `seed/catalog.json` is pinned to Microsoft CDM revision `dd21d715e05ebf740a11356c80b5c3b4c38a89c2`.
3. Install [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/), start Ollama, and follow the [local setup](../README.md#start-locally). To reproduce the current comparison, install `nomic-embed-text:latest` and `gemma3:4b` as well as the models required by the app's `.env`.
4. Select an installed, open-weight chat model that fits the MacStudio for the larger-router run. Record its exact tag, digest, quantization, model size, runtime and hardware, for example with `ollama list` and Ollama's `/api/tags`; inspect `ollama ps` or `/api/ps` **while inference is running** to see the actual runner and memory allocation. Do not infer acceleration from the model name alone. Available models and memory must be checked on the host; they were not verified from the MacBook Air.
5. In the local untracked `.env`, set `CDM_RAG_BENCHMARK_MODEL_MAX_GB` high enough for the selected model's **file size**. The default is 5 GiB and the benchmark rejects installed models above it. The number does not cap runtime memory. Review timeouts and the available memory before starting; retain the same embedding model (`nomic-embed-text:latest`) for paired comparison.
6. Start the app with `./scripts/run_local.sh`. The app creates its local persistent index on first use and listens on `127.0.0.1:8124`. Inspect available models and case IDs at `GET /api/v1/retrieval-benchmark/options`. The benchmark pages show fixed snapshots and **do not update automatically** when a new run completes.

The datasets and previous result records are tracked in the repository under `evaluation/`; the pinned catalog and upstream license are under `seed/`. Do not copy private `.env` files or local query logs into shared storage.

## Run the 15 added cases

Run the baseline router first (`gemma3:4b`), then repeat with the larger installed model by changing `CDM_HANDOFF_ROUTER_MODEL` and rerunning the block. Each invocation writes a timestamped directory under `data/` so the two runs do not overwrite one another. Set the benchmark file-size cap in `.env` and restart the app before choosing a model above the default cap.

```sh
set -o pipefail
export CDM_HANDOFF_ROUTER_MODEL='gemma3:4b'  # replace with the larger installed tag for the second run
export CDM_HANDOFF_RUN_DIR="data/macstudio-adaptive-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$CDM_HANDOFF_RUN_DIR"

python3 - <<'PY'
import json, os
from pathlib import Path

source = json.loads(Path('evaluation/adaptive-retrieval-v2.json').read_text())
out = Path(os.environ['CDM_HANDOFF_RUN_DIR'])
for split in ('development', 'test'):
    ids = [case['id'] for case in source['questions']
           if case['split'] == split and case['id'].startswith('adaptive-v2-')]
    body = {'track': 'adaptive_retrieval', 'split': split,
            'models': ['nomic-embed-text:latest'],
            'query_model': os.environ['CDM_HANDOFF_ROUTER_MODEL'],
            'question_ids': ids}
    (out / f'{split}-request.json').write_text(json.dumps(body, indent=2) + '\n')
    print(split, len(ids), 'cases')
PY

for split in development test; do
  curl --fail-with-body --no-buffer \
    -H 'Content-Type: application/json' \
    --data-binary @"$CDM_HANDOFF_RUN_DIR/$split-request.json" \
    http://127.0.0.1:8124/api/v1/retrieval-benchmark/stream \
    | tee "$CDM_HANDOFF_RUN_DIR/$split-events.sse"
done
```

The benchmark streams progress and emits `retrieval_benchmark_completed` or `retrieval_benchmark_error`; inspect both streams and check each result's `log_persisted`. Completed runs also append to `data/retrieval_benchmark_runs.jsonl`, or to the configured `CDM_RAG_DATA_DIR`. Use `GET /api/v1/retrieval-benchmark/runs/{run_id}` to retrieve a full run. Keep the local run records and their model/runtime metadata together. Do not claim a successful run merely because the HTTP connection opened.

## Compare the results

- Keep the index, embedding model, case IDs, raw first query, source revision, retrieval mode, top-k cutoffs, router prompt, excerpt size, and fusion strategy fixed across router models. Run development first; record any change before evaluating test.
- Compare first search versus the adaptive result on the **same cases**. Report recall@6 and @20, fraction with all labeled evidence, MRR, follow-up requests, errors, paired improvements/ties/regressions, and timing. MRR measures the first relevant result and can miss failures to find the other required passages.
- Compare a generated follow-up query alone against the corresponding first search **only on cases where a follow-up was successfully run**. Those conditional averages do not represent all cases. Inspect `router_messages`, `followup_query`, `first_pass_ids`, and `followup_ids` in each result to distinguish query quality from fusion effects.
- For the mortgage case, compare the raw search, generated query alone, adaptive fusion, and separately evaluated user-written query. Keep the human-written query and gold IDs out of the router prompt. An omitted passage may still be relevant because labels are incomplete.
- Record model tags/digests, available memory, observed runner, index reuse/build time, model residency, warm-up, and latency. The current runner deliberately unloads the router and embedder between stages. If changing residency or measuring a different backend, report it as a different runtime condition; do not attribute latency differences to model quality.
- The tables at `/retrieval-benchmark` now include fixed Mac Studio snapshots in addition to historical results. Save new JSON results and a report with run IDs, case selection, method and caveats before deliberately updating a table. Do not overwrite earlier results.

Once the comparison is complete, judge whether a larger model actually identifies missing schema evidence and whether its retrieval gains justify the cost. If it still mostly restates the question, investigate separately whether targeted query decomposition, longer or better-selected evidence, or fusion changes improve multi-source recall. Avoid changing several factors at once.

## Candidate reformulation challenges

A separate provisional set is available through the adaptive benchmark API. To
select it, use `"adaptive_question_set": "reformulation_challenge_candidate"`
in the request, with `track: "adaptive_retrieval"`. The options endpoint exposes
its case IDs as `adaptive_challenge_questions`; choose IDs for one split at a
time. The existing v2 set remains the default.

These 12 multi-passage questions include cross-entity links, concept comparisons,
traits, inheritance, and model ambiguity. The development and test cases share
schema sources and question families, so this visible, assistant-authored set
is not a blind or independent generalization test. Labels are representative
and incomplete: examine per-case passages and adjudicate plausible alternatives.
The dataset also lists three candidate negative controls for a separate answering
or abstention evaluation; the adaptive scorer does not evaluate them.
