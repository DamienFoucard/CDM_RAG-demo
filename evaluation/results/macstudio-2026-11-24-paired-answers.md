# Mac Studio paired answer experiment — 24 November 2026

> The historical raw run artifacts referenced below are not included in this distribution; these results cannot be fully reproduced from this report alone.

This experiment compares raw and rewritten retrieval feeding the same answer model and judge. The automated judge preferred the rewritten arm in three of four development cases and tied one; both test cases tied. Manual inspection found important omissions and unsupported connections even in answers scored 5/5. The score difference is a useful signal for inspection, not an established improvement in answer correctness.

The router and answer model were `qwen3.5:9b-q4_K_M` (digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`); the judge was `gemma4:12b` (digest `c7597fc90b86aff6965317f37dd5465402284e26382d9b4bf2dfe3ef1491f4b6`). Both arms used `nomic-embed-text:latest`, the same hybrid index, and up to six source-deduplicated passages selected from the top 30. Each answer used its own arm's passages. The development artifact contains the exact queries, candidates, evidence, answers, scores and timings. It reports all four pairs scored, no arm errors, and an operator-supplied repository commit `8fe705ecf61ffc80f007c93633c8fbb78d01b893`; container Git status was unknown and code/data hashes are retained in the artifact.

| Development case | Judge raw → rewrite / 5 | Cited-entity F1 raw → rewrite | Labeled passages in context raw → rewrite | Manual observation |
| --- | --- | --- | --- | --- |
| Group holdings | 5 → 5 | 0.571 → 0.571 | 0/4 → 0/4 | Both name group and customer holdings; the rewrite lists more categories, but neither states the requested explicit links. |
| Card and customer | 2 → 4 | 1 → 1 | 0/3 → 1/3 | The rewrite retrieves the FICard relationship and names Account or Contact. It still implies a connection via the holding role that the cited passages do not establish. |
| Holding details | 4 → 5 | 0.600 → 0.600 | 0/4 → 0/4 | Neither gives the `FinancialHolding.Details` link or the FHAccount deposit/withdrawal definition; the rewrite introduces a customer connection via `CustomerRelationship` that is not established by the supplied holding evidence. |
| Common Account and Contact | 2 → 5 | 0 → 1 | 0/2 → 2/2 | The rewrite finds the explicit relationships and improves on the raw answer's incorrect Address field. It still omits that `parentCustomer` can also reference a Contact, despite the judge's 5/5. |

The listed passage IDs are representative labels, not an exhaustive set of relevant evidence; zero matching IDs does not mean no useful evidence was retrieved. In the card case, both arms scored citation F1 of 1 despite the raw answer's wrong relationship, illustrating why entity overlap does not establish claim support. In the common Account case, the rewrite correctly identifies the primary Contact and Account as one parent target, but the source explicitly names **both Account and Contact** as possible targets.

The raw arm's mean retrieval and answer times were 0.347 s and 9.16 s per case. The rewrite arm's mean rewrite, retrieval and answer times were 6.25 s, 0.464 s and 13.17 s; judge time was separate. These sequential times include observed loading/unloading conditions and are not a controlled throughput comparison. The report's automated judgment delta is +1.5/5 over the four paired cases; it does not adjust for the judge errors described above.

## Test cases

The test artifact reports both pairs scored without errors and the same recorded model, index, source and code hashes as development. The judge scored the two arms equally in both cases (0 improved / 2 tied / 0 regressed; mean difference 0/5). The specific answers differ materially:

| Test case | Judge raw → rewrite / 5 | Cited-entity F1 raw → rewrite | Labeled passages in context raw → rewrite | Manual observation |
| --- | --- | --- | --- | --- |
| Customer and partner | 3 → 3 | 0.400 → 0.500 | 0/3 → 1/3 | The rewrite finds the common Contact definition, but neither context contains the labeled common Account definition. The raw answer cites Banking Contact for a common Contact claim; the rewrite associates the Account description with the Contact citation. Neither gives the reference's Account definition fully. |
| Card holding roles | 2 → 2 | 0.667 → 0.500 | 0/3 → 1/3 | Raw names `FICard.Cardholder` and explicitly says role details were unavailable. Rewrite finds the `financialHoldingRole` passage, but incorrectly says that attribute identifies the cardholder, omitting `FICard.Cardholder`. |

The test raw arm averaged 0.332 s retrieval and 8.06 s answer generation. The rewrite arm averaged 6.21 s query rewrite, 0.473 s retrieval and 11.40 s answer generation, excluding judging. Both arms were scored; their equal judge scores do not imply equal factual quality or coverage.

The challenge dataset is source-derived, synthetic, small, published, and not independently reviewed. Its labels and compact reference answers are provisional and incomplete. Development and test questions share sources and families and do not form a blind independent holdout. A separate human review of claims and evidence is needed before drawing broader conclusions.
