# GPU claim-review pilot — 25 November 2026

This diagnostic asks whether larger models improve the **optional serving-time claim reviewer**, which is disabled in the GPU application. It does not evaluate the separate answer-scoring judge used by some benchmarks, nor end-to-end answer quality.

Five original drafts previously identified as having unsupported or misattributed claims were replayed with their original questions and evidence. Four previously reviewed, supported drafts served as controls. The model is not told the expected decisions or the manual observations. The controls' original graph contexts were not saved and are empty here. The cases are correlated (including two controls sharing a question and evidence), assistant-reviewed, and not independently verified. Flagging a draft for an unrelated issue is not a successful diagnosis of the recorded problem. The sample is too small to estimate production error rates.

The [self-contained fixture](../gpu-claim-review-fixtures-v1.json) regenerates the saved prompts, including source passage text. The outputs capture each exact request and structured-output schema, model digest and quantization, per-case decisions, raw responses, elapsed time, and Ollama token timing and memory information. The GPU was an NVIDIA RTX A4000 with 16,376 MiB VRAM on a shared host. Each model ran in a separate coordinated GPU window, and the independent Ollama instance's other resident models were unloaded as needed. Runs were single passes in fixed case order, and initial loads or changing context lengths can affect timings.

The reviewer uses the serving `review_answer` implementation at revision `ddc3373`. The 9B and 14B Qwen runs used temperature 0, thinking disabled, a maximum of 2,048 output tokens, and a per-request context estimate capped at 8,192 tokens; their complete messages, schemas, and options match. This raises the output budget from the serving reviewer's 600 tokens. The configured GPU reviewer tag `qwen2.5:0.5b` was disabled; the 9B row is a stronger direct reference, not a claim that 9B was the active production reviewer. Both 0.5B and GPT-OSS were tested with the serving defaults (thinking disabled, 600 tokens). GPT-OSS was also tested with an experimental `low` reasoning setting and 2,048 tokens, which requires a configuration change and is not the setting currently used by the serving reviewer.

| Reviewer | Known-problem drafts flagged / 5 | Supported controls accepted / 4 | Errors / 9 | Mean elapsed per call | Peak Ollama GPU allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 2.5 0.5B Q4_K_M, configured reviewer defaults | 1 (unrelated reason) | 4 | 3 | 5.21 s | 0.50 GiB |
| Qwen 3.5 9B Q4_K_M | 0 | 4 | 0 | 8.49 s | 5.43 GiB |
| Qwen 3 14B Q4_K_M | 0 | 4 | 0 | 9.13 s | 9.61 GiB |
| GPT-OSS 20B MXFP4, serving defaults | 0 (five errors) | 4 | 5 | 11.94 s | 11.87 GiB |
| GPT-OSS 20B MXFP4, low reasoning | 3 | 4 | 0 | 9.10 s | 11.87 GiB |

The 0.5B reviewer rejected one draft for an unrelated reason while marking its annotated unsupported claim as supported. It accepted a second problematic draft and errored on three. Both larger Qwen reviewers accepted all five known-problem drafts. Under serving defaults, 20B returned empty final answers for all five problematic drafts; errors are not counted as successful flags. With low reasoning it identified the specific annotated support problem in three cases, but accepted the other two. All four controls were accepted in each run. This pilot does not justify enabling any of the reviewers as a reliable safeguard. The serving application currently retains the original answer when its optional review fails, so a model error must not be mistaken for a successful review.

Full recorded runs: [0.5B](gpu-claim-review-05b-serving-2026-11-25.json), [9B](gpu-claim-review-9b-2026-11-25.json), [14B](gpu-claim-review-14b-2026-11-25.json), [20B with serving defaults](gpu-claim-review-20b-serving-2026-11-25.json), and [20B with low reasoning](gpu-claim-review-20b-low-2026-11-25.json).

### Targeted high-reasoning follow-up

After observing the low-reasoning results, a separate exploratory run selected the two missed problematic drafts and two distinct supported controls. GPT-OSS 20B used `high` reasoning and a 4,096-token output budget. For one problematic draft, it used all 4,096 output tokens and produced no final answer after 72.02 seconds; this was budget exhaustion, not a timeout. For the other, it returned `accept` after 60.17 seconds. It accepted both controls. The four-case mean was 42.02 seconds; the first call included model loading. Because the cases were selected after seeing earlier results, this is not a separate held-out accuracy estimate. The [full targeted record](gpu-claim-review-20b-high-targeted-2026-11-25.json) includes the exact requests, decisions, per-call timings, and model digest.

## Reproduce

Use a dedicated model instance and a quiet execution window: model unloading affects every client of that Ollama instance. Install the dependencies from the repository, install the model tags under their exact names, then run, for example:

```bash
python scripts/compare_gpu_reviewers.py \
  --fixtures evaluation/gpu-claim-review-fixtures-v1.json \
  --judge qwen3:14b --base-url http://127.0.0.1:11434 \
  --hardware 'NVIDIA RTX A4000 16 GB, shared host' \
  --output evaluation/results/my-review-run.json
```

The tool refuses to overwrite an existing result and checkpoints after each case. `--reasoning-effort` is opt-in, and `--max-output-tokens` and `--context-cap` record explicit inference settings. Repeat `--case CASE_ID` to select cases for a targeted diagnostic; the saved high-reasoning record lists its selected IDs. The 9B recorded run was constructed directly from the original saved MacStudio inputs using an earlier form of the same harness; the checked-in fixture regenerates its nine prompts exactly.
