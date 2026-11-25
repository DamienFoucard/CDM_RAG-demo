# Graph retrieval benchmark v2: trainable embedding fit times

This report records a local relationship-retrieval run with the trainable GCN
implementation now used by `/retrieval-benchmark`. The saved run is
`1183555b-fe79-42ef-b4b3-d98d0b9fb651` in
`data/retrieval_benchmark_runs.jsonl`.

## Setup

- Run date: 2026-11-23.
- CDM catalog: pinned source commit `dd21d715e05ebf740a11356c80b5c3b4c38a89c2`;
  64 entities, 387 observed directed edges, and 73 relationship labels.
- Evaluation: all 12 development questions from
  `evaluation/relationship-questions-v1.json`.
- Text embedders: `nomic-embed-text:latest` and `qwen3-embedding:0.6b`.
- Selected ranking methods: text hybrid, trainable GCN with one/two/three
  layers, TransE, TransH, and BoxE.
- Total run time: 12.36 seconds. The text feature indexes were reused from disk;
  index preparation (the cached load) took 0.004 seconds for Nomic and 0.002
  seconds for Qwen. Earlier uncached builds for the same catalog and model
  digests took 10.549 seconds for Nomic and 24.417 seconds for Qwen.
  Query-path warm-up took 550.12 ms and 839.32 ms, respectively.
- Ollama reported the `llamacpp` runner for both text embedders. These are
  observations from one local run, not a controlled hardware comparison.

## Embedding build and fit times

The app measures index build time while embedding the graph's edge, node, and
relationship-label texts. It measures graph fit time through parameter
initialization, training, and scoring the observed catalog edges. GCN fitting
uses text-derived node features, so it runs once per selected text embedder.
TransE, TransH, and BoxE depend only on the catalog and are fitted once per
benchmark run, then shared by the text embedder batches. Cached index load time
is shown separately from the original cold index build time.

| Text embedding model | GCN · 1 layer | GCN · 2 layers | GCN · 3 layers |
| --- | ---: | ---: | ---: |
| `nomic-embed-text:latest` | 0.936 s | 0.964 s | 1.084 s |
| `qwen3-embedding:0.6b` | 1.093 s | 1.231 s | 1.093 s |

| Graph embedding method | Fit time per run |
| --- | ---: |
| TransE | 0.424 s |
| TransH | 0.590 s |
| BoxE | 2.226 s |

For Nomic, summing its cold text-index build (10.549 s), all three selected
GCN fits (2.984 s), and the three shared graph fits (3.240 s, counted once)
gives **16.772 s** for the full selected-method setup. Qwen's corresponding
standalone total is **31.073 s**. In a paired run that selects both text
embedders, the shared TransE, TransH, and BoxE fits are paid once, not once per
embedder. The latest paired run itself only loaded the cached Nomic index in
0.004 s and Qwen's in 0.002 s.

Text-index preparation, query warm-up, and per-question retrieval latency are
separate measurements. Fit times are not included in the retrieval latency
columns below.

## Retrieval results

The scores below aggregate the same 12 development questions. Edge recall and
complete-edge rate are measured at 20 results; first-edge MRR is the reciprocal
rank of the first labeled edge. Latency is the mean warm per-question local
retrieval time, including query embedding. ★ marks the overall configuration
leader by first-edge MRR @ 20.

| Text embedding model | Method | Build / fit time | Edge recall @ 20 | All edges @ 20 | First-edge MRR @ 20 | Mean latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Nomic ★ | Text hybrid | 10.549 s cold index build | 100.0% | 100.0% | 1.0000 | 28.08 ms |
| Nomic | GCN · 1 layer | 0.936 s | 100.0% | 100.0% | 0.7569 | 28.22 ms |
| Nomic | GCN · 2 layers | 0.964 s | 100.0% | 100.0% | 0.7917 | 28.22 ms |
| Nomic | GCN · 3 layers | 1.084 s | 100.0% | 100.0% | 0.6556 | 28.21 ms |
| Nomic | TransE | 0.424 s | 100.0% | 100.0% | 0.7708 | 28.22 ms |
| Nomic | TransH | 0.590 s | 100.0% | 100.0% | 0.8125 | 28.21 ms |
| Nomic | BoxE | 2.226 s | 100.0% | 100.0% | 0.7438 | 28.21 ms |
| Qwen 0.6B | Text hybrid | 24.417 s cold index build | 100.0% | 100.0% | 0.8049 | 57.11 ms |
| Qwen 0.6B | GCN · 1 layer | 1.093 s | 100.0% | 100.0% | 0.7236 | 57.32 ms |
| Qwen 0.6B | GCN · 2 layers | 1.231 s | 100.0% | 100.0% | 0.7867 | 57.30 ms |
| Qwen 0.6B | GCN · 3 layers | 1.093 s | 100.0% | 100.0% | 0.6507 | 57.29 ms |
| Qwen 0.6B | TransE | 0.424 s | 91.7% | 91.7% | 0.7014 | 57.30 ms |
| Qwen 0.6B | TransH | 0.590 s | 91.7% | 91.7% | 0.7361 | 57.30 ms |
| Qwen 0.6B | BoxE | 2.226 s | 100.0% | 100.0% | 0.6310 | 57.29 ms |

## Interpretation and limits

For this catalog and configuration, fitting each GCN took about one second;
BoxE took the longest of the three graph embedding baselines at about 2.23
seconds. The additional structural rankings did not improve on Nomic's text
hybrid first-edge MRR in this particular development run. Qwen's TransE and
TransH variants missed one question's labeled edge at the cutoff. The small
sample does not support a general ranking of methods.

The graph methods train on the complete observed CDM graph, including edges
that appear as evaluation labels. This measures retrieval of known graph edges,
not held-out link prediction or discovery of missing relationships. The
questions and labels were written from the pinned CDM source and have not been
independently reviewed; unlisted edges are unjudged. Use the test split only
after choosing methods on development, and treat results as provisional until
the labels receive independent review.
