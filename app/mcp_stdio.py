"""Launch the CDM RAG MCP server over stdio, backed by a local or hosted API."""
from __future__ import annotations

import os
import sys

from app.mcp_server import HTTPBackend, create_server


def main() -> int:
    try:
        backend = HTTPBackend(os.getenv('CDM_RAG_MCP_BASE_URL', 'http://127.0.0.1:8124'),
                              os.getenv('CDM_RAG_API_TOKEN', ''))
    except ValueError as exc:
        print(f'Cannot start CDM RAG MCP: {exc}', file=sys.stderr)
        return 2
    create_server(backend).run(transport='stdio')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
