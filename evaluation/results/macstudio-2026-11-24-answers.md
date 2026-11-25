# Mac Studio fixed-evidence answer comparison — 24 November 2026

> The historical raw run artifacts referenced below are not included in this distribution; these results cannot be fully reproduced from this report alone.

The run completed six questions for each of three answer models, with no reported model or judge errors. On the benchmark's composite score, Qwen 3.5 9B scored 89.0/100, Qwen 3 4B scored 87.0, and Gemma 4 12B scored 86.1. **This small automated comparison does not establish a winner**: the six references are provisional, the judge is also one of the tested models, and the scores include one extractive fallback.

The complete event stream contains the answers, evidence, judgments and per-question metrics. Run ID: `c9946449-4984-41fc-91bd-eda4c9b832f6`; total elapsed time 147.85 s; persistence reported successful. The test uses `nomic-embed-text:latest`, disables reranking and retrieves once per question, giving all three answer models the same selected passages. The candidate pool contained every expected entity for each question; the selected contexts covered a mean 0.833 of expected entities.

| Answer model | Mean answer/reference cosine | Mean cited-entity F1 | Mean judge / 5 | Composite / 100 | Mean generation s | Extractive fallbacks | Observed runner | Peak runtime-reported model bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `qwen3:4b` | 0.8030 | 0.8333 | 4.50 | 87.0 | 8.71 | 1 | llamacpp | 3,181,631,242 |
| `qwen3.5:9b-q4_K_M` | 0.9087 | 0.8833 | 4.33 | 89.0 | 4.26 | 0 | llamacpp | 4,780,353,125 |
| `gemma4:12b` | 0.9169 | 0.8333 | 4.17 | 86.1 | 4.20 | 0 | mlx | 10,721,542,084 |

All three models had six judge scores, zero reported judge errors and four exact cited-entity matches. The runtime's model-size observation is not total machine RAM or an independently measured peak. Generation times include the run's observed model load and residency conditions; they are not a controlled throughput test. The judge was `qwen3:4b` on `llamacpp`.

The composite equally weights normalized answer/reference cosine, cited-entity F1, and normalized judge score. Cosine measures semantic similarity rather than factual correctness. Citation F1 measures overlap with expected entity IDs rather than whether each cited passage supports a claim. The 4B model's customer-links response was an **extractive fallback**, triggered when it failed to produce valid source labels; it was nevertheless judged and included in all of its aggregate scores. Several other 4B outputs contain lengthy, repetitive reasoning-like text, including the loan and common Account answers, despite receiving high judge scores.

The two multi-entity cases exposed a shared evidence limitation. For the funds comparison, the selected context omitted `FIStandingOrder`, although the candidate pool contained it. For customer links, it omitted the separate common `Account` and `Contact` entity passages, although both were in the candidate pool. Thus model answers cannot be assessed independently of the fixed context. The context limit itself is selected from the number of expected entities, which uses reference metadata; this should be considered when interpreting the benchmark.

The judge deserves particular scrutiny: in the funds case it scored the 12B answer 4/5 while its own rationale says that two of the three requested concepts were omitted. This conflicts with the benchmark rubric's stated maximum of 3 for missing a central fact. Its score for the 4B fallback likewise does not mean the system generated a complete answer. Human review of the saved evidence, citations and responses is needed.

The set contains four single-entity questions and two multi-entity comparisons, authored by an assistant and source-checked but not independently reviewed. The pinned CDM source revision is `dd21d715e05ebf740a11356c80b5c3b4c38a89c2`. These fixed-evidence runs compare answer models; they do not measure the effect of query reformulation on answer quality.
