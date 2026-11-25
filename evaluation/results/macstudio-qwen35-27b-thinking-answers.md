# Qwen 3.5 27B thinking: fixed-evidence answer review

This is a post-unblinding manual review, with one attempt per question/arm. The eight selected cases use the same frozen passage hashes as the prior 9B/12B fixed-evidence comparison. The 27B run uses thinking enabled, a 4,096-token output limit, a 16,384 context cap and temperature 0; the 9B/12B/14B tests used no thinking and 700 output tokens, while GPT-OSS 20B used low reasoning and 2,048. These are not settings-controlled comparisons. A failed generation or application fallback is not counted as a supported or unsupported model answer.

## Per-attempt assessment

- **Card/customer, raw — model error; no answer to score.** After 290.769 seconds the application recorded `ModelError: The local model returned an empty answer`. This is an error, not an evidence-aware abstention. The saved raw passages contain a card entity and a role description but not cardholder target types or proof that a cardholder and a holding customer are the same record. On the same evidence 9B, GPT-OSS 20B and LFM2.5 inferred unsupported connections; 14B acknowledged that the relationship was not established.

- **Card/customer, rewrite — model error; no answer to score.** After 280.602 seconds the application recorded the same empty-answer ModelError. The cited evidence would support FICard.Cardholder → Account or Contact [R3] and CustomerFinancialHolding.FinancialHolding → FinancialHolding [R1], but does not establish that the two customers are the same record. Prior 9B and LFM2.5 answers overstated that identity; the 14B and 20B answers did not explicitly equate the records but were incomplete.

## Interrupted run

The eight-arm run was manually stopped after these two errors; its saved status is `interrupted`. The other six cases were **not completed** and have no scores. Accordingly this run cannot show whether 27B thinking improves factual answers on any of the eight cases. Empty outputs alone do not establish why generation failed. A later [captured reasoning-stream follow-up](macstudio-qwen35-27b-reasoning-stream-followup.md) used the same rewrite-arm question and evidence with a changed prompt and context settings, and completed with a supported answer; it is not an exact replay.

## Higher-budget follow-up attempts

Two separate targeted card/holding-role rewrite trials were attempted with an 8,192-token output budget and 16,384 context cap. The first was stopped to change the first-content wait; the second, with a 900-second first-content wait, was stopped at the user's request. Both saved artifacts have `interrupted` status and **zero completed rows**, so neither yielded a model answer to evaluate. The later trial may have overlapped a page test, making any timing interpretation especially uncertain.

The earlier empty answers and these interrupted trials do not establish a confirmed timeout, circular reasoning, or exhaustion of a token budget. The saved client output does not retain the model's reasoning or token counters, so the cause cannot be determined from these artifacts. The experiment does not establish whether the 27B model would solve the hard cases given a completed answer.
