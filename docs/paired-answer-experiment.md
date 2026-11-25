# Paired answer experiment

The standalone `scripts/benchmark_paired_answers.py` compares raw question retrieval against a rewrite-only query on the same questions, catalog, embedding model, answer model, judge and fixed context limit. Each arm searches independently and gives the answer model its own passages. It records the queries, complete candidate lists, context, answers, citations, judge rationales, errors, timings, model metadata and index identity. It counts judge-score gains only for pairs where both arms were scored. It does not change the existing benchmark snapshots or graph embedding benchmark.

`evaluation/paired-answer-challenge-candidate-v1.json` has six multi-entity cases adapted from the published adaptive retrieval set. References were checked against the listed passages. Labels and references are provisional, incomplete and not independently reviewed; the published test cases are not blind. A model judge is fallible; inspect evidence and answers directly, especially apparent gains. The reference entity count is not used to select context.

Passage selection defaults to `--context-selection one_per_source`, matching the
earlier paired experiment. For a separately labeled comparison, use
`--context-selection two_kinds_per_source` to allow up to two passages of
distinct kinds for each entity or catalog source, in retrieval rank order.
Both arms use the chosen policy and the same six-passage cap. The selected
policy is recorded in the run artifact. Changing this policy changes the
experimental condition; do not compare its scores as if only the query changed.

The script calls Ollama directly. It does not acquire the host model-work lease or share the application's model lock. Follow the host admission process and ensure the serving application has no active model calls throughout the run. The process creates or reuses the local index and unloads models between stages. Use installed model tags and a fresh output filename, for example:

```sh
uv run python scripts/benchmark_paired_answers.py \
  --dataset evaluation/paired-answer-challenge-candidate-v1.json \
  --embedding nomic-embed-text:latest \
  --router YOUR_ROUTER_TAG --answer YOUR_ANSWER_TAG --judge YOUR_JUDGE_TAG \
  --output data/paired-answer-challenge-run.json
```

For the Docker image, which does not include Git or `.git`, capture the full source revision on the host and provide it explicitly. Use a container built with this script and dataset, and run only when no other model call is active:

```sh
CDM_SOURCE_COMMIT=$(git rev-parse HEAD)
docker compose exec -T cdm-rag python /app/scripts/benchmark_paired_answers.py \
  --dataset /app/evaluation/paired-answer-challenge-candidate-v1.json \
  --repository-commit "$CDM_SOURCE_COMMIT" \
  --base-url http://host.docker.internal:11434 \
  --data-dir /data --output /data/paired-answer-challenge-run.json \
  --embedding nomic-embed-text:latest \
  --router YOUR_ROUTER_TAG --answer YOUR_ANSWER_TAG --judge YOUR_JUDGE_TAG
```

The supplied commit is recorded as operator supplied and the container's Git status as unknown: a commit alone does not establish that the image came from a clean checkout. The output retains file hashes to identify the actual code and data used. Copy the result from the container's persistent volume before removing it.

Repeat `--question-id ID` to select cases. Run development first; select test separately after the method is fixed. The output is checkpointed after each question and cannot overwrite an existing file. `completed_with_errors` indicates at least one arm failed; inspect each stage and the paired denominator. Results include sensitive query and answer text, so keep the output in local protected storage. The script records the repository commit, dirty status, hashes of its code and key implementation files, dataset and catalog hashes, index revision and Ollama model metadata. Record actual device state and admission externally as well.
