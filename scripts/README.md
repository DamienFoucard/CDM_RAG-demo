# Scripts

The scripts in this directory serve different tasks. Run them from the
repository root; most evaluation scripts require a model and input files and
are not needed to use the application.

| Task | Entry points |
| --- | --- |
| Ask a question or check a token | `ask_cdm.py`, `check_api_token.py`; see the [API examples](../examples/README.md). |
| Start the local service | `run_local.sh`; see the [main README](../README.md). |
| Prepare the catalog or evaluation datasets | `fetch_cdm_source.py`, `build_catalog.py`, and `build_*` scripts. |
| Run retrieval, routing, answer or scope experiments | `benchmark_*`, `compare_*`, and `evaluate_retrieval.py`; see the [evaluation guide](../docs/benchmarking/README.md). |
| Review or export experiment results | `judge_*`, `audit_*`, `review_saved_claims.py`, and `export_*` scripts. |
| Replay or diagnose a particular experiment | `replay_*`, `capture_reasoning_stream.py`, and `diagnose_gpu_rerank_reasoning.py`; see the associated reports in [evaluation results](../evaluation/results/). |

Similar names do not necessarily mean versions of the same tool. For example,
`judge_web.py` judges retrieval evidence while `judge_web_answers.py` separately
grades answers to out-of-scope questions. The current grounding judge is
`judge_grounding_v3.py`. The older v2 script is available in
[repository history](https://github.com/DamienFoucard/CDM_RAG-demo/blob/f72588dd2884bfbae86cdcfd9a26fbd09ac7b289/scripts/judge_grounding_v2.py)
for historical inspection; it uses a different rubric and its checked-in hash
does not match the recorded hash for the saved v2 results, so it is not a
verified way to reproduce those results. Saved historical results remain in
the repository.
