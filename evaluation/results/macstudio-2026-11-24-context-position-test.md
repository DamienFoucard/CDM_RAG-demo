# Context-position diagnostic — 24 November 2026

> The historical raw run artifacts referenced below are not included in this distribution; these results cannot be fully reproduced from this report alone.

GPT-OSS 20B returned the exact target names and supporting passage label in all nine attempts. LFM2.5 24B did so in eight; in the remaining attempt it named only Account when the passage also mapped to Contact. Qwen 3.5 9B met the exact short-name and label criteria in seven; one of the two strict misses identified both correct targets using longer paths. One Magistral 24B pilot returned invalid JSON, so it could not be scored. These are small, source-derived lookup tests, not a measurement of end-to-end RAG or a general long-context ranking.

## Method

The [test script](../../scripts/benchmark_context_distractors.py) selects three previously visible CDM schema questions and their saved source passages. Each question is asked three times with the same supporting passage placed early, in the middle, and late among the same unique distractor passages from the saved CDM catalog. The distractor pool is filtered for terms associated with the target relationship. It is not a random sample of all possible distractors and is not an independently held-out dataset. The model receives the question and passage text; the expected answer and citation are not included in its instructions.

Each generated response must be valid structured JSON and contain the complete exact set of short entity or attribute names and the correct supporting passage label. Case and optional brackets around the label are normalized. This strict exact-label rule makes invalid JSON or a wrong citation an unsuccessful test, even if some useful text was produced. It does not assess the quality of a full natural-language answer. No retrieval is performed.

| Question | Total evidence text | Distractor text | Passages | Expected items |
| --- | ---: | ---: | ---: | --- |
| FICard.Cardholder targets | 23,341 characters | 23,059 characters | 26 | Account, Contact |
| Contact.parentCustomerId targets | 22,950 characters | 22,352 characters | 24 | Account, Contact |
| CustomerRelationship fields mapping to both targets | 23,634 characters | 22,770 characters | 26 | customerId, partnerId |

These sizes count passage text, not tokens, and exclude message framing. Exact consumed input token counts were not recorded. The runtime allocated 11,264 context slots for GPT-OSS and Qwen and 13,312 for LFM2.5, based on the client’s character estimate and output budget; allocated slots are not a measurement of actual input tokens or proof of full long-context coverage. The three complete runs have identical saved evidence hashes for each of the nine matching question/position combinations. The single Magistral pilot has the same hash as the corresponding middle-position case. GPT-OSS and LFM share script and helper hashes; the later runs use an updated harness with optional settings, and their matching evidence hashes are verified separately.

## Results

| Model and inference settings | Early | Middle | Late | Exact items and citation | Mean time per attempt |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 9B Q4, thinking off, 2,048-token output limit | 1/3 | 3/3 | 3/3 | 7/9 | 12.37 s |
| GPT-OSS 20B, low reasoning, 2,048-token output limit | 3/3 | 3/3 | 3/3 | 9/9 | 8.88 s |
| LFM2.5 24B-A2B Q4, thinking on, 4,096-token output limit | 3/3 | 2/3 | 3/3 | 8/9 | 12.14 s |
| Magistral Small 24B Q4, thinking on, 4,096-token output limit | not run | invalid JSON on one pilot | not run | not scored | 40.88 s (pilot) |

All nine LFM responses selected the expected citation. On the middle-position Contact.parentCustomerId case, it cited the correct passage but returned Account alone and omitted Contact. On the early-position version of that case, Qwen cited the correct passage and returned `Account.cdm.json/Account` and `Contact.cdm.json/Contact`; these identify both targets but failed strict matching against the requested short names. Qwen's other miss was an empty item list and citation on the early CustomerRelationship question. On the single Magistral pilot for CustomerRelationship fields, the client reported `ModelError: The local model returned invalid JSON`; the saved artifact does not preserve a valid response to judge, and this is not evidence that Magistral cannot find the answer under another output format.

All three complete runs used temperature 0 and one attempt per case/position; the reasoning modes, output limits and allocated contexts differ. The three positions of each question share the same source material and are correlated, and the questions and reference passages were known during test construction. These figures therefore do not support a general performance ranking, isolate the effect of model size, or establish performance near any model’s advertised maximum context. The reported times are single-run observations that include variable model loading and residency effects, not controlled throughput measurements.

The local detailed artifacts contain source text, model tags and digests, settings, per-case outputs and timings: Qwen, GPT-OSS, LFM2.5, and Magistral pilot. These data files are local and are not included in a clean checkout.

## Separate claim-review format pilots

The optional claim reviewer uses a different prompt and output schema from the lookup test. On one known-problematic draft, LFM2.5 returned an empty answer with thinking enabled at 2,048- and 4,096-token output limits, and again at an 8,192-token limit with an explicit 16,384-token context cap. The earlier attempts used the default role-based context cap. On the same draft, Magistral returned invalid JSON with thinking enabled, a 4,096-token output limit and a 16,384-token context cap. Separately, Qwen 3.5 9B with thinking on and a 2,048-token limit returned empty answers on the first two problematic drafts; its five-case run was then interrupted, leaving the artifact marked `running`. These are failed or interrupted format/budget configurations, not accepted or rejected claims and not a basis for judging factual accuracy. The saved errors do not establish why each output was empty or invalid.
