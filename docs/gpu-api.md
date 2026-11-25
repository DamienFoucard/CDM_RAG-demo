# Use the GPU API from a terminal

The independent CDM RAG GPU service accepts API questions over HTTPS. Use the
included Python client to authenticate with an application token, follow the
request, and read the final answer. It uses the streaming API and does not
require a separate installation of a shell command.

## Before you start

On a trusted machine with Python 3 and this repository checked out, change to
the repository directory. The examples below use a private `.env` in that
directory. It needs `CDM_RAG_GPU_PUBLIC_HOST` and
`CDM_RAG_GPU_INITIAL_API_TOKEN`; `CDM_RAG_API_TOKEN` can be used for a different
application token. Environment settings take precedence over file settings.
Keep the file private and do not commit it or put the
token in a command or URL. Signed-in owners can create and revoke separate
application tokens on the service's **API tokens** page.

The client reads a file only when explicitly selected; if yours is elsewhere,
replace `.env` in each command with its path. It can also use environment
variables without a file. To verify the token without submitting a question or
loading a model:

```sh
python3 scripts/check_api_token.py --env-file .env
```

`Token accepted by CDM RAG (HTTP 200).` means the authenticated status request
succeeded. A rejected token returns HTTP 401 or 403. The check does not prove
that a query will succeed.

## Ask an easy question

```sh
python3 scripts/ask_cdm.py --env-file .env \
  'What does FIOverdraft describe?'
```

The answer should describe an overdraft as credit associated with a checking
account that can allow withdrawals after the balance reaches zero. Inspect the
cited source; the wording and output can vary. For a recorded answer and an
assessment of its citations, see the [API examples](../examples/README.md#easy-overdraft-definition).

## Ask a multi-hop question

```sh
python3 scripts/ask_cdm.py --env-file .env \
  'A bank needs the schema links from a person or company to something they own and onward to its specific loan or investment type. Which links establish this chain?'
```

This is a challenging evaluation question, not a guaranteed answer. Verify
each claimed link against its cited source. To print the full result, including
source evidence and processing metadata, add `--json`:

```sh
python3 scripts/ask_cdm.py --env-file .env --json \
  'A bank needs the schema links from a person or company to something they own and onward to its specific loan or investment type. Which links establish this chain?'
```

To save that JSON, append `> answer.json`; progress still appears in the
terminal because it is written to stderr. The JSON includes the final answer,
sources and a `processing` object. Check `processing.model_calls` for models
actually invoked; `configured_models` alone does not mean a model was used.
When retrieval runs, `retrieval.reranked` and `retrieval.reranker_model` show
whether reranking ran and which model was used. Some requests can take
deterministic paths or stop before reranking or generation.

The independent GPU Compose profile configures `qwen3-embedding:0.6b` for
embedding, `qwen3:4b` for scope checks and query rewriting, and `qwen3.5:9b`
for routing decisions, answers and reranking. Reranking is on by default; the
optional answer reviewer is disabled. Persisted settings may change the
non-embedding models, and a request need not use every configured role. Full
API clients can override the reranking default with the `rerank` boolean;
demo clients may turn it off but cannot enable it if the server default is off.

## What to expect

The client prints stage and model updates as the request runs, then prints the
final answer. A connection-alive message means the connection remains open; it
does not prove a model has produced new output. The client does not display
provisional streamed answer tokens. Models may need to load, so a request can
take minutes. The default `--timeout` is 600 seconds of connection inactivity,
not a limit on total request time.

The client verifies HTTPS, does not print the token, and does not follow
redirects with credentials. Questions and answers are saved in the service's
query log, as with the web application. Citations locate evidence; they do not
guarantee that every claim is supported. See the [streaming API guide](api-streaming.md)
for event and metadata details, or the [MCP guide](mcp.md) for MCP clients.
