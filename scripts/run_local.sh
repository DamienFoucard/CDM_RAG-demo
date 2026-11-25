#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

uv_args=(run --python 3.12)
if [[ -f .env ]]; then
  uv_args+=(--env-file .env)
fi

exec uv "${uv_args[@]}" uvicorn app.main:app --host 127.0.0.1 --port 8124
