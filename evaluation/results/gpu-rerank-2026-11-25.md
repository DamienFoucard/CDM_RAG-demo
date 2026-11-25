# GPU reranking evaluation, 2026-11-25

This exploratory evaluation compares the deployed passage reranker prompt with raw hybrid retrieval using identical saved candidates. It does not exercise routing, question rewriting, or final answers. No production settings were changed. Inputs and per-case outputs are saved alongside this report.

Hardware was an NVIDIA RTX A4000 (16,376 MiB VRAM) on a shared host, using the standalone service's Ollama runtime and the existing `qwen3-embedding:0.6b` index. The index was copied before searching so the experiment did not modify the serving index. Each held-out question produced 20 hybrid candidate chunks once; all reranking models received those same candidates in the same order, using the deployed `app.rag.rerank` prompt and deterministic temperature. Qwen models and the initial GPT-OSS arm used the deployed 300-token limit and `think: false`; a second GPT-OSS arm used `think: low` and a 2,048-token limit. Model context size followed the existing prompt-size policy. Timing covers the reranking call only, excludes initial search, routing and answers, and is measured on a shared host. The first case may include model loading; subsequent averages should not be interpreted as guaranteed warm latency.

The entity dataset contains 24 synthetic, source-checked but not independently reviewed test questions; labels are examples, not exhaustive judgments. Entity metrics deduplicate entity IDs among the first six chunks, so six chunks may represent fewer than six entities. The object dataset contains 13 similarly provisional held-out test questions covering attributes, relationships, traits, inheritance, manifests and catalog passages. It has exact passage-ID targets, including multiple targets for two questions; recall is the fraction of listed targets retained. Candidate-pool metrics cover all 20 input chunks and show whether reranking could possibly recover each target. Unlisted sources are not necessarily irrelevant. Metrics are not measures of grounded answer quality, and no statistical or general-production claim is warranted from these small sets.

Model digests were `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90` (Qwen3.5 9B), `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8` (Qwen3 14B), and `f38aa0c53da5f8c49d08c43a99df24ff53167fe68e24664a7777288e7656fdfe` (GPT-OSS 20B). The embedding digest is recorded with each snapshot. The 14B model was observed at 100% GPU with a 7,168-token context; 20B was also observed at 100% GPU with a 7,168-token context (about 12.5 GB used). This observation does not establish full GPU offload for every larger context.

## Held-out object passages

All 13 questions had exactly 20 input candidate chunks. Twelve of the 13 questions had all listed targets in that pool; the standard-inheritance target was absent. The raw first six chunks retained all listed targets on 9/13 questions (mean target recall 0.692, MRR 0.404). All three models retained 11/13 (mean target recall 0.846), including the manifest-structure and inheritance-family targets that were outside the raw first six. All three missed the account-inheritance target even though it was in the candidate pool, and could not recover the absent standard-inheritance target.

| Reranker | Settings | Passage recall at 6 | MRR at 6 | Mean after first case | Errors |
| --- | --- | ---: | ---: | ---: | ---: |
| None | Raw hybrid | 0.692 | 0.404 | See search timing below | 0 |
| Qwen3.5 9B | App settings | 0.846 | 0.756 | 9.04 s | 0/13 |
| Qwen3 14B | App settings | 0.846 | 0.808 | 10.72 s | 0/13 |
| GPT-OSS 20B | App settings | Not comparable | Not comparable | 14.82 s | 11/13 |
| GPT-OSS 20B | Low reasoning, 2,048-token limit | 0.846 | 0.731 | 12.52 s | 0/13 |

The first calls took 8.24 s, 40.16 s, 107.11 s and 12.41 s respectively, with varying model residency and context lengths. The 20B app-settings score is intentionally omitted: only two of 13 calls returned valid rankings, so scoring only successful calls would be misleading. On three sampled failures, a separate diagnostic with the original settings generated 300 tokens of reasoning and stopped at the length limit without final content. All three produced valid rankings with `think: low` and the larger token limit, retaining their labeled target(s). Both settings changed together, so this does not isolate the effect of the token allowance from the effect of the reasoning setting. The [diagnostic](gpu-rerank-2026-11-25-gpt-oss-diagnostics.json) contains completion counts and final rankings, not reasoning text.

## Held-out entity questions

The raw top six chunks included the labeled entity for 23/24 questions; the remaining label was absent from the full 20-chunk pool. Qwen3.5 9B kept all 23. Both larger models dropped one additional raw hit, on different questions. The 14B model dropped `GroupFinancialHolding` on `test-03`; the 20B model dropped `applicationCommon::Account` on `test-11`. All calls returned valid rankings.

| Reranker | Labeled-entity recall at 6 | MRR at 6 | Mean after first case | Errors |
| --- | ---: | ---: | ---: | ---: |
| None | 0.958 | 0.816 | See search timing below | 0 |
| Qwen3.5 9B | 0.958 | 0.917 | 10.17 s | 0/24 |
| Qwen3 14B | 0.917 | 0.917 | 14.77 s | 0/24 |
| GPT-OSS 20B, low reasoning | 0.917 | 0.917 | 13.11 s | 0/24 |

The first-case times were 29.55 s, 13.47 s and 11.34 s respectively; the first case need not be a cold load. GPT-OSS was tested here only with low reasoning and the 2,048-token cap, after the unchanged settings failed on 11/13 object questions. All results are isolated reranking comparisons, not tests of the subsequent evidence-selection or answer-generation steps.

## Reproduction

With the pinned catalog, an isolated copy of the existing index, and the specified models available at `OLLAMA_BASE_URL`, create a candidate snapshot and run each reranker against it:

```sh
python scripts/benchmark_gpu_reranking.py snapshot \
  --dataset evaluation/catalog-object-retrieval-v1.json \
  --data-dir /path/to/copied/index --snapshot /path/to/object-candidates.json
python scripts/benchmark_gpu_reranking.py rank --model qwen3.5:9b \
  --snapshot /path/to/object-candidates.json --output /path/to/9b.json
python scripts/benchmark_gpu_reranking.py rank --model gpt-oss:20b \
  --reasoning low --max-output 2048 \
  --snapshot /path/to/object-candidates.json --output /path/to/20b-low.json
```

Use `evaluation/cdm-schema-evaluation-v2.json` to create the separate entity snapshot. The default `rank` settings reproduce the serving reranker settings; the GPT-OSS low-reasoning settings are experimental and are not configured in the service. Test runs should be scheduled to avoid contention: direct calls to Ollama do not acquire the serving app's request lock. On the actual runs, mean candidate-search time was about 3.64 s per question for each snapshot, including the configured embedding-model residency behavior. This is not included in the rerank-only timing above and should not be generalized to every deployment.

## Per-question comparisons

Cells show reciprocal rank / target recall. “Pool” is recall among the input 20 chunks; all other columns use the first six chunks. Full questions, labels, candidate IDs, selections, per-call timings, and errors are in the accompanying JSON files.

### Object questions

| Case | Pool recall | Raw RR / recall | 9B RR / recall | 14B RR / recall | 20B low RR / recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| object-test-attribute-phone-number | 1.000 | 0.333 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| object-test-relationship-primary-contact | 1.000 | 0.500 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.500 / 1.000 |
| object-test-trait-application-account-classification | 1.000 | 0.500 / 1.000 | 1.000 / 1.000 | 0.500 / 1.000 | 1.000 / 1.000 |
| object-test-trait-definition-default | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| object-test-inheritance-account | 1.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| object-test-inheritance-base-standard | 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| object-test-common-manifest-overview | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| object-test-foundation-manifest-declarations | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| object-test-manifest-structure-application-common | 1.000 | 0.000 / 0.000 | 0.500 / 1.000 | 1.000 / 1.000 | 0.500 / 1.000 |
| object-test-manifest-relationship-owner-user | 1.000 | 0.333 / 1.000 | 0.333 / 1.000 | 1.000 / 1.000 | 0.500 / 1.000 |
| object-test-catalog-coverage | 1.000 | 0.333 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| object-test-trait-definitions-display-and-description | 1.000 | 0.250 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| object-test-inheritance-activity-family | 1.000 | 0.000 / 0.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

### Entity questions

| Case | Pool recall | Raw RR / recall | 9B RR / recall | 14B RR / recall | 20B low RR / recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test-01 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-02 | 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| test-03 | 1.000 | 0.500 / 1.000 | 0.500 / 1.000 | 0.000 / 0.000 | 1.000 / 1.000 |
| test-04 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-05 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-06 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-07 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-08 | 1.000 | 0.250 / 1.000 | 0.500 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-09 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-10 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-11 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.000 / 0.000 |
| test-12 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-13 | 1.000 | 0.333 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-14 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-15 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-16 | 1.000 | 0.500 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-17 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-18 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-19 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-20 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-21 | 1.000 | 0.500 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-22 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-23 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| test-24 | 1.000 | 0.500 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
