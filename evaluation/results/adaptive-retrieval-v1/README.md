# Adaptive retrieval comparison: 24 November 2026

These are saved results from the `adaptive_retrieval` benchmark on an Apple M1 MacBook Air with 8 GiB of unified memory. The fixed embedding model was `nomic-embed-text:latest` (F16, digest prefix `0a109f422b47e3a3`); the follow-up router was `gemma3:4b` (Q4_K_M, digest prefix `68ceab7e468a31c9`). Both runs reused the same Nomic index; index preparation was excluded from per-question timing. Query embedding was unloaded after each use and the router unloaded after each decision, so times include observed loading and unloading on this host.

| Split | Strategy | MRR@6 | Recall@6 | Hit@1 | Mean time | Follow-up requests | MRR improved / unchanged / worsened |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | First search | 0.6667 | 0.8333 | 0.625 | 0.56 s | — | baseline |
| Development | Adaptive | 0.7708 | 0.8750 | 0.750 | 12.87 s | 4 / 8 | 1 / 7 / 0 |
| Test | First search | 0.6250 | 0.7083 | 0.500 | 0.57 s | — | baseline |
| Test | Adaptive | 0.5625 | 0.5833 | 0.500 | 11.57 s | 4 / 8 | 0 / 7 / 1 |

The adaptive approach improved rank for the development account identifier question and retrieved an additional labeled entity for the development funds comparison. On the test team-membership question, fusion moved the labeled entity outside the top six. These outcomes describe the tracked labels, which are incomplete; they are not a judgment of answer quality.

The development and test splits each have eight published, synthetic, source-checked cases; test is not blinded. The source dataset and provenance are in [`adaptive-retrieval-v1.json`](../../adaptive-retrieval-v1.json). The complete paired [development](development.json) and [test](test.json) records include decisions, queries, ranks, errors, timings, and run IDs. Gold labels were not given to the router. The experiment does not use the production planner, scope gate, answer generator, or judge. A second search is fused with the first using reciprocal-rank fusion; on routing or follow-up errors the first results are retained.
