# Judge feedback and bounded retry pilot

This is a small live pilot of the interview API, not an independently labeled
quality benchmark. The full observations are in
[`results/judge-feedback-v1/pilot.json`](results/judge-feedback-v1/pilot.json)
contains the complete saved observations.

## What changed

Generated answers can receive one structured review. The judge checks cited
claims and question coverage, then accepts, asks for a revision from the same
evidence, requests one targeted retrieval, or abstains. A revision must pass a
second review. An unavailable judge leaves the original citation-validated
answer intact. Out-of-scope requests and verified graph answers skip the judge.

## What the live pilot found

- The 9B reviewer accepted the definition and core-attribute answers, adding
  about 8.5 seconds and 16.3 seconds respectively in these runs.
- The 4B reviewer accepted the same core-attribute answer with about 5.5
  seconds of judge time. It reuses the answer model, so it is a correlated,
  non-independent check. Review is opt-in by default because this pilot did not
  establish a quality gain to justify that latency.
- After switching the local demo default off, the same live query returned its
  cited attribute answer with zero judge calls. Its 17.6-second request time is
  not directly comparable to the earlier 10.9-second judge-enabled sample;
  model warm-up and request timings were uncontrolled.
- The 9B reviewer accepted a relationship answer that reversed the actual
  Email→Account and Activity→Account edges. Its claim checks marked both claims
  supported. This is a concrete failure, so the service now answers qualified
  relationship-list questions directly from the directed graph. The same
  question then returned six correctly oriented edges and source entity
  citations in about six seconds.
- The qualified Contact/Account graph question returned both directions in
  about 2.2 seconds. A current-balance request and a dog story were stopped
  before retrieval and made zero judge calls.

These observations do not establish that judge-and-retry improves answer
quality overall. The judge adds latency and can miss relationship-direction
errors; it remains a bounded check for generated definitions and attributes,
while relationship answers use the deterministic graph path.

## Tests and limits

The focused RAG suite passed 39 tests. The available repository suite passed
59 tests when excluding two optional graph-benchmark modules that
require the uninstalled `torch` dependency. Unit tests cover same-evidence
revision, targeted re-retrieval, accepted retries, rejected retries, judge
errors, and abstention.

The live set is small and assistant-authored. Timings were not controlled for
model warm-up, and no independent human labels were applied. A live retrieval
retry was not observed in this pilot; that path is covered by deterministic
tests.
