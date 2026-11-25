# Application packages

The HTTP entrypoint remains `app.main:app`; the MCP stdio entrypoint remains
`python -m app.mcp_stdio`.

- `schema/` contains catalog extraction and graph projection.
- `retrieval/` contains indexing, routing, search, and answer orchestration.
- `evaluation/` contains the evaluation implementations used by the API.
- `runtime/` contains model access and request observability.
- `security/` contains API token and demo-access controls.

The Python modules previously at the root of `app/` now live in these
packages. For example, `app.rag` is `app.retrieval.rag`, and `app.corpus` is
`app.schema.corpus`. Update imports and monkeypatch targets accordingly. This
reorganization does not change the HTTP API or the two entrypoints above.
