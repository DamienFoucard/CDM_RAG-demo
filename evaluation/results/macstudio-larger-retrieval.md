# Larger retrieval routers on Mac Studio — 24 November 2026

> The historical raw run artifacts referenced below are not included in this distribution; these results cannot be fully reproduced from this report alone.

Gemma 4 12B had mixed results relative to Qwen 3.5 9B on the same 12 hard cases. It improved average labeled recall at six passages on the six test cases, but lost recall at twenty and took longer. A one-case Gemma 4 31B pilot recovered the same single labeled relationship as the smaller models; it did not find the other two. These are retrieval results, not generated-answer scores.

All runs use the `reformulation-challenge-candidate-v1` source-derived dataset, the same pinned Microsoft CDM catalog revision `dd21d715e05ebf740a11356c80b5c3b4c38a89c2`, Nomic Embed Text, the same hybrid index `cd491dd029b4b4bc1e13e924ab8f1f66d1447f938fd03f70dcd4cd9a8d96775f`, and the same router instructions and fusion algorithm. The first search uses the question as written. The router sees its top six passages, may request one more search, and reciprocal-rank fusion combines two top-20 rankings. The router does not see labels. These runs test the experimental follow-up track, not automatic follow-up in the ordinary question endpoint.

## Full 9B and 12B runs

Each split contains six cases; all four runs completed with `log_persisted: true`, no router errors and no unload warnings. Recall is the average fraction of labeled passages found; “all at 6” counts cases where all labeled passages appear in the first six. Mean time includes first search, router, unload, and any second search; it is specific to observed model residency and host state.

| Split | Router | First → adaptive recall@6 | First → adaptive recall@20 | Adaptive MRR@6 | Adaptive all at 6 | Follow-ups | Mean adaptive time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Development | Qwen 3.5 9B | 0.319 → 0.375 | 0.569 → 0.736 | 0.436 | 1/6 | 4/6 | 8.56 s |
| Development | Gemma 4 12B | 0.319 → 0.319 | 0.569 → 0.625 | 0.394 | 1/6 | 5/6 | 13.30 s |
| Test | Qwen 3.5 9B | 0.444 → 0.478 | 0.644 → 0.844 | 0.367 | 1/6 | 3/6 | 7.56 s |
| Test | Gemma 4 12B | 0.444 → 0.533 | 0.644 → 0.789 | 0.492 | 2/6 | 4/6 | 11.89 s |

Against its own first search, the 12B follow-up improved/tied/worsened recall@6 on **1/4/1** development and **3/2/1** test cases. At twenty passages, its counts were **1/5/0** and **3/3/0**. On test, 12B recovered all three labeled localized-trait passages within six, versus two for 9B. On the group-asset case 12B found two of three labels within twenty, versus three for 9B. Both found none of the three loan-portfolio labels within six after fusion. On development, 12B lost the one credit/payment label that the initial search had placed within six. More follow-up searches therefore did not consistently yield more useful top results.

Run records: development 9B `f02ce112-d0ec-419f-9f70-af10115ca356`; development 12B `d60505cd-1ab9-4681-b9fd-a026f851a3bb`; test 9B `282e7046-7c8f-4619-a651-d39184cb7a13`; test 12B `85506cbf-1d98-403a-ac69-e55c86edb448`.

## 31B single-case pilot

The 31B pilot, run `c466e9ff-8f5d-4a9e-b299-47346f9c89ad`, completed and persisted with one follow-up, no reported router error and no unload warning. The only case was `reformulation-challenge-dev-customer-holding`: identify links from a person or company to a holding and onward to a loan or investment type. It has three labeled passages:

1. `RetailBankingCoreDataModel::CustomerFinancialHolding:relationship:1` — `Customer` to Account or Contact.
2. `RetailBankingCoreDataModel::CustomerFinancialHolding:relationship:2` — `FinancialHolding` to FinancialHolding.
3. `RetailBankingCoreDataModel::FinancialHolding:relationship:1` — `Details` to holding types.

The raw first search found **0/3** within both six and twenty. All three routers' follow-up searches found only the third passage, at follow-up rank 1. After fusion its rank was 5 for 9B, 6 for 12B, and 5 for 31B. The two CustomerFinancialHolding links remained outside the top 20 for all three: **1/3 labeled passages at both cutoffs**, not a complete chain. Unlabeled passages may still be relevant, but cannot by themselves establish that these three labeled links were retrieved. The 31B adaptive stage took **41.50 s** versus **15.22 s** for 9B and **18.17 s** for 12B on this case; these individual timings include model load/unload and are not throughput benchmarks.

During the 31B run, Ollama reported approximately **19 GB resident**, **100% GPU**, `llamacpp` and **5,120 context tokens**. MacOS reported about **70% free memory** before, **15% then 9%** during loading/inference, and **73%** immediately after unloading. Swap in use was **21,765.69 MiB before**, **30,211.50 MiB loaded**, and **28,192.75 MiB after**. These are spot observations, not measured peaks. The long-run effect on other shared-host workloads was not measured, and one pilot does not support a 31B quality estimate.

## Model and context provenance

The installed tags and observed digests were `qwen3.5:9b-q4_K_M` (`56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`), `gemma4:12b` (`c7597fc90b86aff6965317f37dd5465402284e26382d9b4bf2dfe3ef1491f4b6`) and `gemma4:31b-q4_0` (`54f997c79a7f22f5273ab67a2e569687d218c02f1103445d4a8e26a5fc6ce30f`). Nomic's digest was `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`. Digests identify local artifacts; the 31B is a Q4 quantization.

The earlier client chose a context cap from whether a model tag also happened to be configured as an answer or judge model. Under the recorded application configuration, reconstruction from the saved router messages shows that the 9B and 12B runs requested **5,120 tokens for the first development and third test case, and 4,096 for the other cases**. The 31B pilot, run with the new uniform 8,192-token cap, requested 5,120 for that same development case. The cap is a maximum; requested context still follows the same character-based heuristic. This establishes equivalent requested windows for those comparisons, not proof that different tokenizers processed every input token without truncation. The earlier 4B baseline used a 4,096-token cap on the two longer cases and should not be treated as having an identical window.

The cases are small, synthetic or source-derived, published, not independently reviewed, and have incomplete passage labels. Development and test share sources and question families. The LLM-suggested query may include guessed names, and fusion may push useful evidence down. Inspect actual retrieved passages and source definitions before concluding that an answer is supported. Neither passage recall nor this pilot establishes answer correctness or performance on unseen questions.
