# Relationship retrieval pilot

This is a comparison of evidence retrieval, not a claim about answer accuracy.
All five configurations use the same pinned Microsoft/CDM snapshot
`dd21d715e05ebf740a11356c80b5c3b4c38a89c2`: 64 extracted entities, 387 directed
reference edges, and 73 distinct relationship-attribute labels. The active
serving index and application were not changed.

## Questions and labels

The [main set](relationship-questions-v1.json) contains 12 development and 24
test questions, including six two-hop test questions. Development and test
have disjoint `(source entity, relationship attribute)` pairs. Every test
question names at least one of its labeled source entities, so these cases
favor entity-name matching. The separately generated
[exploratory challenge](relationship-challenge-v1.json) contains 16 paraphrased
questions, including four two-hop questions, with no gold source entity named
verbatim. It was authored after looking at the development results, and some
of its edges also occur in the main set; it is not an independent holdout.

An assistant authored all questions from the extracted graph. Each example
directed edge was checked against the graph and its pinned source file, but
the questions and relevance labels have not had independent human review.
Labels are incomplete: other returned edges may also be useful. There are no
negative or unanswerable cases in this pilot.

## Methods

- **Hybrid, no rerank:** the prototype's vector and BM25 retrieval and entity
  expansion, without the optional LLM reranker.
- **Hybrid + rerank:** the current prototype's retrieval, `qwen3:4b` passage
  reranking, explicit-name handling and expansion of up to four entities.
- **Neighborhood + rerank:** the same passage IDs and hybrid search, with up
  to 12 fixed, typed, directional graph references appended to each passage
  before embedding and reranking. This uses an isolated index.
- **Graph assisted:** begin with the reranked baseline's entities, add names
  matched directly in the question, traverse at most two hops, and rank
  *observed* edges using lexical similarity, the existing text embeddings and
  proximity. Return at most 20 edges.
- **Graph + TransE:** the same graph-assisted candidates and score, plus a
  weighted TransE triple-score rank. The 32-dimensional model was trained with
  an L1 margin-ranking loss and same-model corrupted triples. It saw all 387
  observed edges but no query labels. This is retrieval over a known graph,
  **not** held-out link prediction; unobserved triples are not treated as
  facts. The model's zero final sampled training loss is not an accuracy score.

All methods receive the raw question. The prototype's similarity gate, query
planner and final answer generation are excluded. The graph and neighborhood
methods are benchmark implementations, not enabled in the app. Timings are
warm local single runs, with other local services running; they include the
LLM reranker where shown but exclude end-to-end API and answer time. The two
graph methods' shared timing includes ranking both variants.

## Results

“All edges” counts questions for which every example gold edge was present.
“All @20” applies the same test to the first 20 returned edges. For two-hop
questions, both labeled edges must be present; this does not measure whether an
answer correctly explains the path. MRR rewards the position of the **first**
labeled edge, so it alone does not establish a complete path.

The baseline expands full entity contexts and does not explicitly rank the
edges inside them; its @20 and MRR scores use their existing flattened order.
The graph methods explicitly rank edges and cap the result at 20. Thus their
ordering and evidence budgets differ: the @20/MRR comparison measures the
effect of prioritizing edges, not an equal-budget comparison of answer quality.

Main test, 24 synthetic questions:

| Configuration | All edges | All @20 | First-edge MRR@20 | Mean edges | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid, no rerank | 24/24 | 16/24 | 0.355 | 48.6 | 0.10 |
| Hybrid + rerank | 24/24 | 16/24 | 0.355 | 39.6 | 3.79 |
| Neighborhood + rerank | 24/24 | 17/24 | 0.357 | 43.6 | 7.77 |
| Graph assisted | 24/24 | 24/24 | 0.958 | 19.4 | 3.92 |
| Graph + TransE | 24/24 | 24/24 | 0.847 | 19.4 | 3.92 |

On the six two-hop questions, both labeled edges appeared in the full evidence
for every configuration. At 20 edges, the baseline and neighborhood variants
contained both for 2/6 questions; the two graph variants did so for 6/6.

Exploratory paraphrase challenge, 16 synthetic questions:

| Configuration | All edges | All @20 | First-edge MRR@20 | Mean edges | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid, no rerank | 13/16 | 9/16 | 0.203 | 36.3 | 0.09 |
| Hybrid + rerank | 13/16 | 9/16 | 0.203 | 44.5 | 4.66 |
| Neighborhood + rerank | 14/16 | 10/16 | 0.265 | 31.3 | 7.58 |
| Graph assisted | 14/16 | 14/16 | 0.707 | 19.5 | 4.78 |
| Graph + TransE | 14/16 | 14/16 | 0.708 | 19.5 | 4.78 |

On its four two-hop questions, every configuration returned both labeled edges
somewhere for 2/4 cases. At 20 edges, the baseline and neighborhood variants
did so for 0/4; both graph variants did so for 2/4. The graph variants missed
the questions about a membership-to-organization path and a customer's monetary
unit-to-organization path. These are useful examples to inspect, not evidence
that the unlabeled returned edges are irrelevant.

Compared directly with graph-assisted ranking, adding TransE improved the
first labeled edge's reciprocal rank on 1 main-test question, worsened it on
6, and tied on 17; on the challenge it improved 1, worsened 1 and tied on 14.
It did not change whether all labeled edges appeared in the first 20 on either
set. This is one structural prior and one training configuration, not a verdict
on graph embeddings generally.

The development run built the neighborhood index in 31.56 seconds on this Mac;
subsequent runs reused it. Training TransE took about 2.9 seconds. Those are
observed setup times, not controlled or portable benchmarks. Raw per-case
results and graph edges are in [development](results/relationships-v1/development.json),
[test](results/relationships-v1/test.json), and
[challenge](results/relationships-v1/challenge.json).

## LLM judge

The local judge evaluates a fixed, stratified sample of questions. It receives
a shuffled pool of the top five edges from each method plus the example labels,
with method names hidden. It grades each edge as direct evidence, related
context, or unrelated. The resulting nDCG@5 uses only this pooled set as its
ideal; judgments do not establish answer correctness. The judge is `qwen3:4b`,
also used for reranking, so its preferences are not independent.

On a fixed sample of 12 main-test cases, pooled nDCG@5 was 0.218 for each of
the three expanded-context variants, 0.924 for graph-assisted ranking and
0.810 with the TransE prior. The same ordering and evidence-budget caveat above
applies: the baseline does not rank its edges to optimize the first five.
On a fixed sample of 10 challenge cases, the corresponding scores were 0.336,
0.336, 0.332, 0.763 and 0.740. The challenge judge recorded no zero grades for
the example gold edges; this does not establish that its other grades are
correct. Full pooled judgments and reasons are in the
[challenge judge artifact](results/relationships-v1/challenge-judge.json).

The [main-test judge audit](results/relationships-v1/test-judge.json) contains
one disagreement with an example gold edge. For a path from a queue item's
worker User to that User's Territory, the judge marked `User --territory-->
Territory` irrelevant because it does not itself start at QueueItem. That edge
is the requested second step. The disagreement is retained in the artifact;
the judge is not a substitute for human review.

The saved JSON files contain the question sets, per-case results, model and
code digests, paired comparisons, and judge artifacts for all 15 runs.

## Next evaluation

The [web-question shortlist](web-question-candidates-v1.json) records public
banking FAQs and real forum scenarios, with source links, any changes made to
fit the CDM-schema task, and tentative matching entities or edges. It also
includes questions that require actual records or runtime behavior. This is a
convenience sample, selected and mapped by the assistant after the pilot; it
has not been independently judged or scored and is not an independent test set.

Collect independently authored banking questions and have reviewers judge
multiple relevant edges and paths, including direction, missing links and
ambiguous names. Compare equal context budgets and measure supported answer
quality, citation correctness, latency, and no-answer behavior. Separately
evaluate held-out graph edges if the goal is link prediction; prevent those
edges from leaking through text passages or the graph index.
