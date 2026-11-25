# Live query progress

`POST /api/v1/query` returns the final answer and a `processing` object. To
receive updates while the same work is running, send the same JSON body to
`POST /api/v1/query/stream`. It responds as `text/event-stream` (SSE). For a
local installation:

```sh
curl -N http://localhost:8124/api/v1/query/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does FIOverdraft describe?"}'
```

The independent GPU installation requires a valid bearer token for both API
routes. The `scripts/ask_cdm.py` client loads it from an explicitly selected
private environment file, reports progress to stderr, and prints the final
answer to stdout. `--json` prints the complete final result on stdout, including
sources and processing metadata. `--timeout` sets connection inactivity, not a
limit on the whole run.
For copyable authenticated GPU examples, including a token check and complete
JSON output, see the [GPU API guide](gpu-api.md). The terminal client displays
stage and model events but does not print provisional token text.

SSE events are JSON in `data:` frames:

- `status` has a `stage`, `query_id` and elapsed milliseconds. The first
  `waiting_for_model` event is sent immediately with zero elapsed time; later
  events name the active stage and report elapsed time. The final `queue_ms`
  reports time spent waiting for a worker and the model lock.
- `model` reports the actual model, role and stage when an invocation starts.
- `token` carries provisional answer text. Citation formatting or review can
  change it; the final `result.answer` is authoritative.
- `result` contains the same complete result object as `/api/v1/query`.
  `error` ends the stream when the request cannot complete.

A `: keep-alive` SSE comment shows that the connection is alive. It does not
prove that a model is making progress, and does not claim that new model output
has arrived.

The final result's `processing` object contains:

- `elapsed_ms`: wall time from the request handler, including queueing.
- `queue_ms`: time waiting for the worker and the model-run lock.
- `stage_timings`: completed stages with their duration and status.
- `configured_models`: models configured for the request; a configured role
  is not proof that the model ran.
- `model_calls`: actual embedding and local chat invocations, including role,
  model, stage, and completed or failed outcome. A hosted JEV router call is
  recorded as attempted when its router diagnostics are available.
- `answer_method`: the recorded generator or deterministic verified response;
  `no_generation` applies when no generator is recorded.

A relevance gate may return without routing or generation. A verified catalog or
graph response can bypass the answer model. The optional answer judge and
reranker run only when enabled and applicable. Check `model_calls` and
`answer_method` rather than assuming every configured model was used.
