# Expanded adaptive retrieval comparison · 24 November 2026

This comparison uses the 15 new cases from [`adaptive-retrieval-v2.json`](../../adaptive-retrieval-v2.json): seven development and eight test questions. Twelve require multiple labeled passages. The examples are user-provided or assistant-authored, source-checked, incomplete and not independently reviewed; the test cases are published and are not blind. They cover the pinned Retail Banking and applicationCommon catalog, not the full CDM. The previous 16 cases and [v1 run](../adaptive-retrieval-v1/README.md) are retained separately: v1 used shorter router excerpts and did not evaluate top-20 fusion.

Each case begins with the same raw question and the same top-20 Nomic hybrid search. Gemma 3 4B sees the question, a list of object types and up to six initial passages, each capped at 2,500 characters. It may request one additional search. The adaptive result combines the two ranked lists with reciprocal-rank fusion, or retains the first list on no-search or error. The follow-up query alone is also scored, using only cases where it was actually executed; its baseline is calculated on that same subset. The labels and the user's separate human-written query are never sent to the router. The human query is evaluated independently on one case and is not an assumed optimum.

| Split and cases | Strategy | Cases | Recall@6 | Recall@20 | All labels@6 | All labels@20 | Mean time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development, all | First search | 7 | 0.333 | 0.631 | 0.143 | 0.429 | 0.46 s |
| Development, all | Adaptive + fusion | 7 | 0.357 | 0.607 | 0.143 | 0.429 | 17.38 s |
| Development, follow-up subset | First search | 6 | 0.222 | 0.569 | 0.000 | 0.333 | 0.49 s |
| Development, follow-up subset | Follow-up query alone | 6 | 0.167 | 0.417 | 0.000 | 0.000 | 1.13 s |
| Test, all | First search | 8 | 0.417 | 0.792 | 0.250 | 0.625 | 0.48 s |
| Test, all | Adaptive + fusion | 8 | 0.417 | 0.688 | 0.250 | 0.500 | 17.57 s |
| Test, follow-up subset | First search | 6 | 0.333 | 0.778 | 0.167 | 0.667 | 0.53 s |
| Test, follow-up subset | Follow-up query alone | 6 | 0.333 | 0.639 | 0.167 | 0.500 | 0.96 s |
| Mortgage example | First search | 1 | 0.167 | 0.667 | 0.000 | 0.000 | 0.61 s |
| Mortgage example | User-written query | 1 | 0.500 | 0.500 | 0.000 | 0.000 | 0.33 s |

“Recall” is the fraction of labeled evidence retrieved; “All labels” is the fraction of questions for which every label was retrieved. The two adaptive rows include router/model load, unload and follow-up search time; follow-up-only and human-query times are search times and exclude the time needed to produce the query. Index preparation is excluded. Gemma requested follow-up searches for 6/7 development and 6/8 test cases; no errors were recorded. On test, top-six recall was unchanged for each case, while top-20 recall worsened for two. These scores measure labeled retrieval, not answer quality.

For the mortgage case, raw top-six results included the `financialHoldingRole` description; the user's query additionally surfaced the `Customer` and `FinancialHolding` relationship passages. Neither query retrieved all six labeled passages in its top 20. The router's own follow-up was more generic and did not improve the top-six recall for that case. The catalog includes role values, but does not contain customer records or the full mortgage application entity.

The runs used an Apple M1 MacBook Air with 8 GiB unified memory, Nomic Embed Text F16 (digest prefix `0a109f422b47e3a3`), and Gemma 3 4B Q4_K_M (digest prefix `68ceab7e468a31c9`). Both reused the same index. The embedder was unloaded after each use and the router after each decision. Runtime conditions affect the observed delays.

The complete [development](development.json) and [test](test.json) records include router input, queries, rankings, timing, errors, and run IDs. Ordinary question answering does not use this experimental router.
