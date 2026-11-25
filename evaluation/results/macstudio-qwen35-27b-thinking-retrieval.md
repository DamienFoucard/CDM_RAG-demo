# Qwen 3.5 27B with reasoning: hard retrieval replay

> The historical raw run artifacts referenced below are not included in this distribution; these results cannot be fully reproduced from this report alone.

On the six saved test questions, Qwen 3.5 27B with thinking enabled improved average labeled recall at six over the 9B baseline but did not match the 12B baseline. It placed a labeled passage first on three questions, compared with one for 9B and two for 12B. It was much slower in this local run. On the separate customer-to-holding development question, its final top 20 recovered none of the three labeled chain passages, compared with one for each smaller router.

These are **seven exploratory questions**, not a general ranking. They are published, source-derived cases with provisional, incomplete passage labels; development and test share sources and question families. Earlier models used disabled thinking and a 160-token output budget, while 27B used thinking and up to 4,096 output tokens, so model and inference settings changed together. Recall and ranking are not evidence that an answer is correct.

## Method and provenance

The 27B replay uses the exact first-pass IDs and router messages saved for the 9B baseline, the same Nomic embedding model, the same hybrid index and pinned CDM catalog, and the same follow-up search and reciprocal-rank fusion. It withholds reference labels from the router and uses them only for subsequent scoring. The two saved smaller-model runs have the same first-pass rankings; their original adaptive results are reported below. The replay verifies the catalog revision, index identity, and reconstructed messages before inference. It is an offline adaptive-routing experiment; it does not enable this router in the ordinary question endpoint.

The tested tag is `qwen3.5:27b-q4_K_M`, digest `64ae5b1d651fb2709d5928ccb8dcfb4bb2cc9c2c8001ac34497e36a6cf905b59`. Its settings were `think: true`, temperature 0, a 4,096-token output budget, and a 16,384-token context cap. Actual runtime context windows on the six test cases were 7,168–9,216 tokens. The character-based sizing estimate is not proof that a tokenizer preserved every input token. The index identity was `cd491dd029b4b4bc1e13e924ab8f1f66d1447f938fd03f70dcd4cd9a8d96775f`; the CDM source revision was `dd21d715e05ebf740a11356c80b5c3b4c38a89c2`.

The first test run was **interrupted** by the memory monitor during the third case; only its first two completed rows are included. The remaining four questions were run later with a lower stopping threshold after the user accepted tighter memory. The table combines those six unique completed questions; it does not treat the interrupted third attempt as a result or repeat either completed row. The separately run development pilot is not included in the test averages.

## Six-question test set

Recall is the mean fraction of labeled passages recovered. MRR@6 rewards placing the first labeled passage higher; it does not require finding the whole chain. All@6 is the number of questions with all labeled passages among six. First pass is the shared original search before any follow-up.

| Measure | First pass | 9B adaptive | 12B adaptive | 27B thinking adaptive |
| --- | ---: | ---: | ---: | ---: |
| Mean recall@6 | 0.444 | 0.478 | 0.533 | 0.500 |
| Mean recall@20 | 0.644 | 0.844 | 0.789 | 0.844 |
| MRR@6 | 0.325 | 0.367 | 0.492 | 0.569 |
| Hit@1 | 1/6 | 1/6 | 2/6 | 3/6 |
| All labeled passages@6 | 1/6 | 1/6 | 2/6 | 1/6 |

The 27B router requested a second search for four of six questions. Against its own first pass it improved recall@6 on the group-asset question, tied on the other five, and did not worsen recall@6 on this set. A tie can still change the ordering substantially.

| Test question | First @6 | 9B @6 / @20 | 12B @6 / @20 | 27B @6 / @20 | Observation |
| --- | ---: | ---: | ---: | ---: | --- |
| Group asset | 0/3 | 1/3 / 3/3 | 1/3 / 2/3 | 1/3 / 3/3 | 27B matched 9B at both cutoffs and found one more labeled passage than 12B at 20. |
| Holding role | 2/3 | 2/3 / 3/3 | 2/3 / 3/3 | 2/3 / 3/3 | 27B elected not to search again. |
| Trait comparison | 2/3 | 2/3 / 3/3 | 3/3 / 3/3 | 2/3 / 3/3 | 27B elected not to search again and did not match 12B's complete top six. |
| Instrument options | 0/5 | 1/5 / 2/5 | 1/5 / 2/5 | 0/5 / 2/5 | Follow-up found labeled material, but fusion left none among the first six. |
| Two account types | 2/2 | 2/2 / 2/2 | 2/2 / 2/2 | 2/2 / 2/2 | All labels were already in the first six. |
| Loan portfolio | 1/3 | 0/3 / 2/3 | 0/3 / 2/3 | 1/3 / 2/3 | 27B placed the labeled `FHInvestment` entity first; this was already in the initial six. |

The mean **router stage** was 103.70 seconds for 27B (range 37.00–250.85), compared with 7.05 seconds for 9B and 11.22 seconds for 12B in their earlier runs. These are observed single-run wall times, not controlled throughput benchmarks: models, reasoning budgets, context windows, load and memory state differ. The 27B figure excludes the original search and follow-up embedding/search; earlier saved total adaptive latency was 7.56 seconds for 9B and 11.89 seconds for 12B, which is a different measurement.

## Customer-to-holding development pilot

The question asks for links from a person or company to a holding and onward to a loan or investment type. There are three labeled relationship passages: `CustomerFinancialHolding:relationship:1`, `CustomerFinancialHolding:relationship:2`, and `FinancialHolding:relationship:1`. The first pass found 0/3 at six and at 20. Both earlier 9B and 12B adaptive results found the third passage, giving 1/3 at each cutoff. The 27B follow-up found that passage at follow-up rank 12, but fusion placed it outside the retained top 20; neither of the first two labeled links appeared in its saved follow-up top 20. Its final result was therefore **0/3 at six and at 20**, not a recovered chain. This pilot's router stage took 72.41 seconds.

## Host observations and reproducibility

The short 4,096-context fit check completed with thinking enabled. In sampled observations the system free-memory indicator reached 13%, while swap in use rose from about 16,736 MiB to a peak of 18,868 MiB; Ollama reported approximately 16.67 GB resident on GPU. The development retrieval pilot had a sampled minimum of 12% free. The first test run's guard stopped it at a sampled 10% free and 20,703 MiB swap. The later four-case run completed with a sampled minimum of 11% free and a maximum of 20,896 MiB swap. These are spot samples, not guaranteed peaks; the Mac is a shared host, and these observations do not establish a safe margin for concurrent workloads.

Local artifacts: development pilot, interrupted test run with two completed rows, four remaining test cases, fit monitor, and the interrupted test monitor and retry monitor. These ignored local run artifacts contain saved questions and benchmark records. The replay entry point is `scripts/replay_reasoning_adaptive.py`; it does not reserve a host model-work slot, so it should be run alone with memory monitoring.
