# Experimental graph expansion of hard retrieval cases

> The historical raw run artifacts referenced below are not included in this distribution; these results cannot be fully reproduced from this report alone.

The two missing links in the customer-holding case can be recovered by a bounded walk over explicit schema references. Starting with the saved 9B or 12B adaptive results, this replay brought all three labeled links into the first six, versus one of three before expansion. This is an **offline retrieval replay**, not a new answer benchmark and not enabled in the application. No model was called. Expansion was developed using the visible customer-holding case, and test results were inspected during tuning. Neither split is blind or an independent estimate; the small source-derived dataset has incomplete labels and shared sources.

## Method and limits

The baseline is the exact reciprocal-rank fusion of the saved first and follow-up results for [9B and 12B](macstudio-larger-retrieval.md), verified against each saved run's recall, all-at-k, MRR and hit-at-one. For each question the replay takes entity IDs among the top six adaptive passages as seeds, walks the resolved schema graph at most two hops in either direction, and ranks reachable relationship-bearing entities by the question's lexical overlap with their names, descriptions, relationship text and resolved targets' descriptions. It takes up to three relationship passages from the ranked entities and inserts them after the first three baseline passages, de-duplicates, and retains 20. The relationship passages are existing source-addressable catalog passages; it does not manufacture graph links. The scorer receives gold labels only after expansion. The method does not use prior conversational messages or human reference queries. It does not prove that retrieved passages form a valid path: connectivity and direction still need interpretation.

The method is intentionally general but was adjusted after seeing the development example: for instance, generic suffix normalization was corrected to match “own” and “owned”, and the generic domain word “bank” was excluded. Test output was visible at the time. It can also insert irrelevant relationship passages into trait and inheritance queries. There is no routing gate or guarantee of preserving baseline recall at fixed cutoffs. Accordingly, do **not** enable this experimental method by default. To use it in the product after further validation, expose it as an explicit experiment flag, run expansion after the existing adaptive ranking with a hard budget and lineage retained, and independently evaluate a routing gate and evidence displacement against unseen, manually reviewed cases. The current replay is directly runnable locally:

```bash
.venv/bin/python scripts/replay_graph_expansion.py \
  --output data/macstudio-runs/graph-expansion-12b.json \
  data/macstudio-runs/adaptive-development-12b-events.sse \
  data/macstudio-runs/adaptive-test-12b-events.sse
```

Use the corresponding `9b` files for the other router. Local detailed results, including ranked IDs, seeds, selected graph passages, matching terms and timings: 9B and 12B. The catalog and dataset must match the run's dataset and source revision, and the saved baseline scores must reproduce before replay continues.

## Results

Each split has six questions. The initial rankings use the question directly; adaptive means the existing router and follow-up search; graph means the offline expansion of that adaptive ranking. Recall is the mean fraction of the labeled passages found. All-at-six counts cases for which all labels appear among six. These scores do not measure answer correctness and omit relevant but unlabeled evidence.

| Split | Router | Recall@6 first / adaptive / graph | Recall@20 first / adaptive / graph | All@6 first / adaptive / graph | MRR@6 adaptive / graph |
| --- | --- | --- | --- | --- | --- |
| Development | 9B | .319 / .375 / .625 | .569 / .736 / .792 | 1 / 1 / 3 | .436 / .417 |
| Development | 12B | .319 / .319 / .625 | .569 / .625 / .736 | 1 / 1 / 3 | .394 / .417 |
| Test | 9B | .444 / .478 / .622 | .644 / .844 / .844 | 1 / 1 / 3 | .367 / .375 |
| Test | 12B | .444 / .533 / .789 | .644 / .789 / .844 | 1 / 2 / 4 | .492 / .542 |

At six, the 9B credit/payment case falls from 1/3 to 0/3 and the 9B localized-trait comparison falls from 2/3 to 0/3. At twenty, the 9B common-base case falls from 1/3 to 0/3: injected unrelated relationship passages displaced an inheritance passage. The 12B expansion did not lower labeled recall at these cutoffs on these twelve cases, but that does not establish that its additional passages are useful or its answers grounded.

### Per-case labeled recall

Each router cell shows adaptive → graph; first is the common first-pass score. Per-case values are fractions, not counts of relevant passages.

| Case | First @6 | 9B @6 | 12B @6 | First @20 | 9B @20 | 12B @20 |
| --- | --- | --- | --- | --- | --- | --- |
| dev-customer-holding | 0.000 | 0.333 → 1.000 | 0.333 → 1.000 | 0.000 | 0.333 → 1.000 | 0.333 → 1.000 |
| dev-credit-payment | 0.333 | 0.333 → 0.000 | 0.000 → 0.000 | 0.667 | 1.000 → 1.000 | 0.667 → 0.667 |
| dev-trait-language | 1.000 | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
| dev-card-link | 0.333 | 0.333 → 1.000 | 0.333 → 1.000 | 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
| dev-person-group | 0.250 | 0.250 → 0.750 | 0.250 → 0.750 | 0.750 | 0.750 → 0.750 | 0.750 → 0.750 |
| dev-common-base | 0.000 | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 | 0.333 → 0.000 | 0.000 → 0.000 |
| test-group-asset | 0.000 | 0.333 → 1.000 | 0.333 → 1.000 | 0.333 | 1.000 → 1.000 | 0.667 → 1.000 |
| test-role | 0.667 | 0.667 → 1.000 | 0.667 → 1.000 | 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
| test-trait-compare | 0.667 | 0.667 → 0.000 | 1.000 → 1.000 | 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
| test-instrument-options | 0.000 | 0.200 → 0.400 | 0.200 → 0.400 | 0.200 | 0.400 → 0.400 | 0.400 → 0.400 |
| test-two-accounts | 1.000 | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
| test-loan-portfolio | 0.333 | 0.000 → 0.333 | 0.000 → 0.333 | 0.333 | 0.667 → 0.667 | 0.667 → 0.667 |

For the focal question, the expansion selected, in order: `CustomerFinancialHolding:relationship:1` (Customer → Account or Contact), `CustomerFinancialHolding:relationship:2` (FinancialHolding → FinancialHolding), and `FinancialHolding:relationship:1` (Details → the holding types). These links are in the `RetailBankingCoreDataModel`; source passages retain their original catalog citations. The analogous group-asset case reached all three labels at six for both routers.

## Cost and checks

On this small 64-entity, 387-edge catalog, a single replay built its graph and source groups in about 60 ms. Across the 24 per-case expansions the observed in-process expansion times averaged about 0.07 ms (range 0.025–0.249 ms); 9–55 graph nodes were visited and 12–683 adjacency entries examined per case. These are one-run local CPU timings, not service latency; they exclude the original searches, model inference, I/O, and any production integration. Two-hop traversal is bounded by depth and seed count, but the number of traversed edges is data-dependent and would need a separate hard edge/node budget for larger catalogs.

The replay verifies source revisions and reproduces saved baseline metrics before scoring. Focused tests cover a two-hop path, unresolved references, de-duplication, preserved prefix, and non-entity seeds. The experiment has not been integrated into the ordinary retrieval endpoint or evaluated for downstream answer quality.
