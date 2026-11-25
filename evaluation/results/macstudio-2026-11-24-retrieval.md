# Mac Studio retrieval comparisons — 24 November 2026

> The historical raw run artifacts referenced below are not included in this distribution; these results cannot be fully reproduced from this report alone.

The 9B evidence-informed router improved average labeled passage recall on this small challenge set, particularly at 20 passages, but also pushed one test case's labeled evidence out of the first six. The 4B router requested no follow-up and therefore reproduced the first search on every challenge case. In the separate contextual query experiment, reformulation sometimes improved first-result rank and sometimes lost the labeled result entirely. These results measure retrieval, not answer quality.

Both experiments used `nomic-embed-text:latest` (digest `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`), the same hybrid index (`cd491dd029b4b4bc1e13e924ab8f1f66d1447f938fd03f70dcd4cd9a8d96775f`), and the pinned Microsoft CDM source `dd21d715e05ebf740a11356c80b5c3b4c38a89c2`. The model tags were `qwen3:4b` and `qwen3.5:9b-q4_K_M`. The first contextual run built the index in 74.143 s; later runs reused it. All six streams reported completion and `log_persisted: true`; no benchmark error or unload warning event was recorded.

## Contextual query formulation

Each of four development and four test cases compares the raw question plus prior messages with a planner and a rewrite-only query, with the same first-search index. Reranking and answer generation were disabled. MRR@6 is the mean reciprocal rank of the first labeled result; recall@6 is the fraction of labeled evidence found within six. Mean times include query inference and warm search but exclude index preparation. Each split has only four cases, so p95 and means are descriptive rather than stable latency estimates.

| Split | Query | Recall@6 | MRR@6 | Mean ms | p95 ms | MRR cases improved / tied / worsened vs raw |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Dev | Raw | 1.000 | 0.508 | 64 | 95 | — |
| Dev | Planner 4B | 1.000 | 0.875 | 1,046 | 2,665 | 2 / 2 / 0 |
| Dev | Rewrite 4B | 1.000 | 0.833 | 474 | 549 | 2 / 1 / 1 |
| Dev | Planner 9B | 1.000 | 1.000 | 2,672 | 7,925 | 3 / 1 / 0 |
| Dev | Rewrite 9B | 0.750 | 0.625 | 976 | 1,091 | 2 / 1 / 1 |
| Test | Raw | 0.750 | 0.750 | 60 | 106 | — |
| Test | Planner 4B | 0.500 | 0.500 | 963 | 2,482 | 0 / 3 / 1 |
| Test | Rewrite 4B | 0.500 | 0.500 | 391 | 456 | 0 / 3 / 1 |
| Test | Planner 9B | 0.500 | 0.375 | 2,631 | 8,040 | 0 / 2 / 2 |
| Test | Rewrite 9B | 0.750 | 0.625 | 916 | 1,172 | 1 / 1 / 2 |

Per-case reciprocal rank at six, including zeros for missed labels:

| Case suffix | Raw | Planner 4B | Rewrite 4B | Planner 9B | Rewrite 9B |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev-overdraft-entity | 1 | 1 | 1 | 1 | 0 |
| dev-account-identifier | 0.333 | 1 | 1 | 1 | 1 |
| dev-localized-description-trait | 0.200 | 1 | 1 | 1 | 1 |
| dev-loan-entity | 0.500 | 0.500 | 0.333 | 1 | 0.500 |
| test-contact-entity | 1 | 0 | 0 | 0 | 0 |
| test-phone-call-number | 1 | 1 | 1 | 0.500 | 0.500 |
| test-account-inheritance | 0 | 0 | 0 | 0 | 1 |
| test-deposit-account | 1 | 1 | 1 | 1 | 1 |

All case IDs above start with `context-`. For example, the 9B rewrite recovered the test inheritance label at rank one, while all four generated test queries missed the common Contact label that raw input retrieved first. The latter rewrites shortened away descriptive context. Full requests, queries and rankings: development stream, run `164ac7ee-b4bf-4b93-8522-46b50a8c8b7c`; test stream, run `66b2aa3c-0ff4-49cc-b60d-ebbb47dd302a`.

## Evidence-informed follow-up

The experimental router sees the same question and first six passages and may request one additional search; reciprocal-rank fusion combines the two top-20 lists. Each split has six cases. First and adaptive scores below are paired within each run. Mean times include the initial search, routing, model unload and any second search; they are not directly comparable to the warmed contextual search timings above.

| Split | Router | Recall@6 first → adaptive | Recall@20 first → adaptive | MRR@6 first → adaptive | All labels@6 first → adaptive | Mean ms first → adaptive | Follow-ups / errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dev | 4B | 0.319 → 0.319 | 0.569 → 0.569 | 0.375 → 0.375 | 0.167 → 0.167 | 302 → 3,905 | 0 / 0 |
| Dev | 9B | 0.319 → 0.375 | 0.569 → 0.736 | 0.375 → 0.436 | 0.167 → 0.167 | 315 → 8,558 | 4 / 0 |
| Test | 4B | 0.444 → 0.444 | 0.644 → 0.644 | 0.325 → 0.325 | 0.167 → 0.167 | 301 → 3,609 | 0 / 0 |
| Test | 9B | 0.444 → 0.478 | 0.644 → 0.844 | 0.325 → 0.367 | 0.167 → 0.167 | 333 → 7,563 | 3 / 0 |

The 9B follow-up improved/tied/worsened recall@6 on 1/5/0 development and 2/3/1 test cases; at 20 the counts were 3/3/0 in each split. The 4B results tied all six first searches in each split. The individual 9B recall values are below; the 4B values equal the first column for every case.

| Case suffix | Recall@6 first → 9B | Recall@20 first → 9B |
| --- | --- | --- |
| dev-customer-holding | 0 → 0.333 | 0 → 0.333 |
| dev-credit-payment | 0.333 → 0.333 | 0.667 → 1 |
| dev-trait-language | 1 → 1 | 1 → 1 |
| dev-card-link | 0.333 → 0.333 | 1 → 1 |
| dev-person-group | 0.250 → 0.250 | 0.750 → 0.750 |
| dev-common-base | 0 → 0 | 0 → 0.333 |
| test-group-asset | 0 → 0.333 | 0.333 → 1 |
| test-role | 0.667 → 0.667 | 1 → 1 |
| test-trait-compare | 0.667 → 0.667 | 1 → 1 |
| test-instrument-options | 0 → 0.200 | 0.200 → 0.400 |
| test-two-accounts | 1 → 1 | 1 → 1 |
| test-loan-portfolio | 0.333 → 0 | 0.333 → 0.667 |

All case IDs above start with `reformulation-challenge-`. On only the cases where a 9B follow-up ran, follow-up-only recall@6 was 0.229 versus a matched first-pass 0.146 on four development cases, and 0.578 versus 0.111 on three test cases. These conditional subsets cannot be compared with full-split averages. Streams and run IDs: dev 4B `eec54e81-c8cc-46dc-8b97-d6530de9dcdc`; dev 9B `f02ce112-d0ec-419f-9f70-af10115ca356`; test 4B `98d30540-e0ae-4c89-8a7e-2bbdae9e062f`; test 9B `282e7046-7c8f-4619-a651-d39184cb7a13`.

## Limits

The contextual set and the [challenge candidate](../reformulation-challenge-candidate-v1.json) are synthetic or source-derived, their labels are incomplete and not independently reviewed, and their test cases are published. The challenge splits share schema sources and question families, so they are not an independent blind test. An unlabelled passage may still be relevant; manual inspection is required before treating a ranking change as better evidence. Follow-up results depend on the router prompt, excerpt cap and fusion as well as model size. The ordinary answer path does not automatically use this experimental router. Model loading, unloading, cache and host state affect latency; these comparisons do not prove answer correctness or a production performance target.
