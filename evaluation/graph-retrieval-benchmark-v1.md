# Graph retrieval benchmark v1 (archived baseline)

This is an archived retrieval-only comparison for the relationship track on
`/retrieval-benchmark`. It records the earlier fixed-propagation GCN baseline,
before the app trained GCN weights. Its GCN build times were not measured and
are not comparable with the current trainable GCN, TransE, TransH, and BoxE fit
times. Current runs save method fit costs with their results in
`data/retrieval_benchmark_runs.jsonl`. This report compares two text embedding
models and seven ranking methods on the same CDM edge catalog; it does not
measure generated answers.

## Setup and scope

- Catalog revision: `dd21d715e05ebf740a11356c80b5c3b4c38a89c2`, with 64 entities,
  387 observed directed edges, and 73 relationship labels.
- Development: all 12 cases from `relationship-questions-v1.json`.
- Paraphrase challenge: all 16 cases from `relationship-challenge-v1.json`.
- Text embedders: `qwen3-embedding:0.6b` and `nomic-embed-text:latest`.
- Methods: text hybrid, GCN-style propagation with one/two/three layers, TransE,
  TransH, and BoxE.
- Saved run IDs: development `1bb509b5-fe17-43cf-8018-f229ceb07266`; challenge
  `5fc21cf0-70c2-49c3-9a15-97a2d21bb9e4` (latest complete challenge run).

The text hybrid baseline fuses BM25 and query-to-edge cosine ranks. The GCN
ablation starts with text embeddings of entity names and descriptions, applies
the symmetrically normalized adjacency matrix with self-loops for the selected
depth, and combines the propagated source, relation, and target features. It
uses fixed identity weights and no nonlinearity or supervised training, so the
label in the UI is explicitly **GCN-style propagation**. TransE, TransH, and
BoxE are 32-dimensional NumPy implementations, trained for 80 epochs with a
fixed seed and one same-model corrupted edge per positive edge. Each graph
method adds its edge ranking to the BM25 and text-similarity ranks.

The graph embedding methods see the **complete observed graph**, including
edges used as evaluation labels. These numbers measure retrieval of known
edges, not held-out link prediction or inferred facts. Questions and labels
were assistant-authored and source-checked, but not independently reviewed;
labels are incomplete, and other retrieved edges may also be useful. The
paraphrase challenge was created after the initial pilot and is not an
independent holdout. The main 24-case test split remains unused in these runs.

## Development results

All rows use the 12 development questions. Recall and complete-edge rate are
at 20; MRR is the reciprocal rank of the first labeled edge. Latency includes
the warmed query embedding and ranking, but excludes query warmup, graph
training, and initial text feature preparation.

| Text embedding model | Method | Edge recall @ 20 | All edges @ 20 | First-edge MRR @ 20 | Mean ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3 Embedding 0.6B | Text hybrid | 100.0% | 100.0% | 0.8049 | 50.83 |
| Qwen3 Embedding 0.6B | GCN-style · 1 layer | 100.0% | 100.0% | 0.8292 | 51.11 |
| Qwen3 Embedding 0.6B | GCN-style · 2 layers | 100.0% | 100.0% | 0.8292 | 51.10 |
| Qwen3 Embedding 0.6B | GCN-style · 3 layers | 100.0% | 100.0% | 0.8292 | 51.08 |
| Qwen3 Embedding 0.6B | TransE | 91.7% | 91.7% | 0.7014 | 51.03 |
| Qwen3 Embedding 0.6B | TransH | 91.7% | 91.7% | 0.7361 | 51.03 |
| Qwen3 Embedding 0.6B | BoxE | 100.0% | 100.0% | 0.6310 | 51.03 |
| Nomic Embed Text | Text hybrid | 100.0% | 100.0% | 1.0000 | 23.85 |
| Nomic Embed Text | GCN-style · 1 layer | 100.0% | 100.0% | 1.0000 | 24.01 |
| Nomic Embed Text | GCN-style · 2 layers | 100.0% | 100.0% | 1.0000 | 24.00 |
| Nomic Embed Text | GCN-style · 3 layers | 100.0% | 100.0% | 1.0000 | 24.00 |
| Nomic Embed Text | TransE | 100.0% | 100.0% | 0.7708 | 23.97 |
| Nomic Embed Text | TransH | 100.0% | 100.0% | 0.8125 | 23.97 |
| Nomic Embed Text | BoxE | 100.0% | 100.0% | 0.7438 | 23.97 |

## Paraphrase challenge results

All rows use the 16 exploratory challenge questions. This is a useful harder
set, but the provenance above means it should not be treated as independent
test evidence.

| Text embedding model | Method | Edge recall @ 20 | All edges @ 20 | First-edge MRR @ 20 | Mean ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3 Embedding 0.6B | Text hybrid | 93.8% | 87.5% | 0.5873 | 56.86 |
| Qwen3 Embedding 0.6B | GCN-style · 1 layer | 93.8% | 87.5% | 0.7427 | 57.17 |
| Qwen3 Embedding 0.6B | GCN-style · 2 layers | 93.8% | 87.5% | 0.7322 | 57.13 |
| Qwen3 Embedding 0.6B | GCN-style · 3 layers | 93.8% | 87.5% | 0.7308 | 57.09 |
| Qwen3 Embedding 0.6B | TransE | 93.8% | 87.5% | 0.5524 | 57.05 |
| Qwen3 Embedding 0.6B | TransH | 93.8% | 87.5% | 0.6198 | 57.05 |
| Qwen3 Embedding 0.6B | BoxE | 93.8% | 87.5% | 0.4594 | 57.05 |
| Nomic Embed Text | Text hybrid | 90.6% | 81.3% | 0.7287 | 23.00 |
| Nomic Embed Text | GCN-style · 1 layer | 93.8% | 87.5% | 0.7921 | 23.17 |
| Nomic Embed Text | GCN-style · 2 layers | 93.8% | 87.5% | 0.7926 | 23.16 |
| Nomic Embed Text | GCN-style · 3 layers | 93.8% | 87.5% | 0.7921 | 23.16 |
| Nomic Embed Text | TransE | 90.6% | 87.5% | 0.7786 | 23.13 |
| Nomic Embed Text | TransH | 90.6% | 87.5% | 0.6912 | 23.13 |
| Nomic Embed Text | BoxE | 90.6% | 81.3% | 0.5245 | 23.13 |

The first cache build took 24.42 seconds for Qwen and 10.55 seconds for Nomic;
later runs reused the on-disk vectors. Query warmup is measured separately and
took 0.29–0.85 seconds in the latest runs. The graph embedding trainers take
about 0.4 seconds (TransE), 0.7 seconds (TransH), and 2.2 seconds (BoxE) here.
These are one-machine timing observations, not controlled hardware benchmarks.
The embedder is unloaded after each selected model's batch.

## Reading these scores

The development set is easy for the text baseline: nearly every method gets
all example edges into its first 20 results. On the exploratory paraphrases,
one-hop GCN-style propagation improved first-edge ranking for both text
embedders, while the deeper settings were similar. The structural priors did
not consistently improve edge recall or first-edge ranking. A larger,
independently authored and judged question set is needed before selecting a
default method. Open the retrieval page to inspect saved per-question rankings
and compare additional models or question subsets.
