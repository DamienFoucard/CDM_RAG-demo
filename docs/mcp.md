# Connect an MCP client

CDM RAG includes a first-party Model Context Protocol (MCP) server. It exposes
two read-only tools backed by the existing application:

- `ask_schema_question`: accepts a question of 1–2,000 characters and returns
  a bounded answer and source pointers. The same query logging rules apply as
  for the HTTP API. Citations identify evidence; they are not a guarantee that
  an answer is correct.
- `service_status`: reports a bounded summary of service and catalog status.

The server does not expose API token administration or private runtime settings
as tools. Invalid input, an unavailable service, a timeout and insufficient
evidence are reported to the client. The question tool can take several minutes
while models load and run; its backend timeout is 600 seconds. If a request
times out, the model worker may continue until its current work finishes. An
answer or its evidence that exceeds the MCP response limits is withheld with
`result_limit_exceeded`; use the HTTP API to request the complete result.

## Connect directly over HTTP

Use the **Streamable HTTP** transport with one of these endpoint URLs:

| Profile | MCP endpoint | Authentication |
| --- | --- | --- |
| Local default | `http://127.0.0.1:8124/api/v1/mcp/` | Local profile disables API authentication by default; keep the service private. If enabled, provide a token. |
| Independent GPU service | `https://cdm-rag.example.com/api/v1/mcp/` | Supply a CDM RAG API token as `Authorization: Bearer <API token>`. |

Replace the example hostname above with your own service host. Create a separate token for each application on the signed-in **API tokens**
page. Provide the token through your client's protected credential settings and
do not put it in a URL or commit it to a repository. The hosted server uses a
manually provisioned bearer token; it does not provide OAuth discovery. The
client must support supplying the header itself.

## Launch over stdio

Run `uv sync` in the CDM RAG checkout and start the local service, or point the
launcher at the hosted service. The local launcher defaults to
`http://127.0.0.1:8124`:

```sh
uv run python -m app.mcp_stdio
```

An MCP client starts this process and communicates with it over standard input
and output. Use the Python environment where the project is installed. For
example, a client with an `mcpServers` JSON configuration can launch the local
profile as follows; replace `/absolute/path/to/CDM_RAG` with your checkout:

```json
{
  "mcpServers": {
    "cdm-rag": {
      "command": "/absolute/path/to/CDM_RAG/.venv/bin/python",
      "args": ["-m", "app.mcp_stdio"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/CDM_RAG",
        "CDM_RAG_MCP_BASE_URL": "http://127.0.0.1:8124"
      }
    }
  }
}
```

To use the stdio launcher with the GPU deployment, set
`CDM_RAG_MCP_BASE_URL=https://cdm-rag.example.com` and supply
`CDM_RAG_API_TOKEN` using the MCP client's protected environment or secret
store. A token is required for a remote HTTPS backend and for any local backend
with API authentication enabled. Remote backend origins must use HTTPS; the
launcher does not follow redirects with credentials.

See the [official MCP specification](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
for transport details and your MCP client's documentation for its precise
configuration format.
