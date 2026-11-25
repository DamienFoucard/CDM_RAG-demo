# Testing CDM RAG

The automated suite checks application behavior using small, repeatable
fixtures and simulated dependencies. It does not need the GPU service, a live
API token or a running Ollama model. From the repository root, install the test
dependencies and run the suite:

```sh
uv sync --frozen --extra test
uv run pytest -ra
```

The `-ra` option explains any skipped tests. Node.js is needed for the
dependency-free JavaScript graph and evaluation-page checks. A small number of
historical benchmark checks need local run artifacts that are not included in
this repository; those checks skip when the artifacts are unavailable. The
offline TransE experiment also skips unless its optional PyTorch dependency is
installed.

## Explore the test suite

| Area | Examples | What is checked |
| --- | --- | --- |
| Retrieval and evidence | `tests/test_retrieval.py`, `tests/test_graph_retrieval_boundaries.py`, `tests/test_evidence_selection.py` | Search results, source-qualified graph references, and selection of supporting passages. |
| Answers and citations | `tests/test_rag.py`, `tests/test_graph_edge_citations.py`, `tests/test_entity_definition_response.py` | Grounded answer paths, source references, and graph-based responses. |
| Routing | `tests/test_query_router.py`, `tests/test_decision_model_settings.py` | Routing choices and model configuration. |
| API and progress | `tests/test_main.py`, `tests/test_query_processing.py`, `tests/test_stream_client_contract.py` | Request handling, streaming progress, result metadata and the terminal client. |
| Authentication | `tests/test_api_tokens.py`, `tests/test_demo_access_integration.py`, `tests/test_check_api_token.py` | Token validation, owner and demo permissions, expiry, and the token-checking client. |
| MCP | `tests/test_mcp_server.py`, `tests/test_mcp_documentation.py` | Protocol initialization, tools, authentication and the integration guide. |
| Graph interface | `tests/test_graph_visibility.py`, `tests/test_graph_fullscreen.py` | Graph filtering and full-screen behavior. |

For a smaller local run, specify one or more files:

```sh
uv run pytest -ra tests/test_retrieval.py tests/test_graph_edge_citations.py
uv run pytest -ra tests/test_api_tokens.py tests/test_mcp_server.py
```

The CI workflow also builds the application image with Docker. Locally, you can
run the equivalent build after the tests:

```sh
docker build --tag cdm-rag-standalone:validation .
```

Passing tests verifies the covered behavior; it does not establish that every
generated answer is correct. The separate [evaluation guide](benchmarking/README.md)
describes measured retrieval and answer quality, datasets and limitations.
