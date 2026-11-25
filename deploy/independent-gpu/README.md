# Independent GPU deployment

This deployment contains only the standalone CDM RAG app, a dedicated Ollama,
a new Keycloak instance and PostgreSQL database, an OAuth2 proxy and a private
Nginx router. The public TLS terminator forwards the CDM RAG host
to `127.0.0.1:19089`. Each service binds only to a separate host loopback
port; none listens on a public interface. Local host processes can reach the
backend directly and therefore must be trusted. The independent
Keycloak realm permits only explicitly provisioned users and disables signup.
Browser pages require login. API calls can use an authenticated owner session or
an application bearer token; the hosted MCP endpoint requires a bearer token.
Concurrent sign-in attempts use separate short-lived CSRF cookies.

This is a Docker Compose deployment, not a separate Kubernetes cluster. It
shares the physical GPU with other host workloads. There is no cross-service GPU
scheduler: arrange an exclusive execution window before loading or running a
model, monitor other workloads and GPU memory, and stop if the hardware is busy.
Do not change the existing Docker daemon or unrelated applications.

## Provisioning

1. On the host, create a protected deployment directory outside shared releases.
   Copy this directory there; create a separate owner-only `.env` (mode `0600`).
   Supply these values in it; do not commit or print the populated file:

   - `CDM_RAG_PUBLIC_HOST` — the dedicated HTTPS hostname.
   - `CDM_RAG_IMAGE` — standalone product image pinned as `repository@sha256:...`.
   - `CDM_RAG_DB_PASSWORD` — independent database password.
   - `CDM_RAG_ADMIN_USERNAME`, `CDM_RAG_ADMIN_PASSWORD` — independent Keycloak
     bootstrap administrator credentials, distinct from the application user.
   - `CDM_RAG_OWNER_USERNAME`, `CDM_RAG_OWNER_EMAIL`, `CDM_RAG_OWNER_FIRST_NAME`,
     `CDM_RAG_OWNER_LAST_NAME`, `CDM_RAG_OWNER_PASSWORD` —
     independent application login. Keycloak marks this email verified because
     no mail provider is installed.
   - `CDM_RAG_OIDC_CLIENT_SECRET` — random shared secret for this new client.
   - `CDM_RAG_COOKIE_SECRET` — random 16-, 24- or 32-byte base64 cookie secret.
   - `CDM_RAG_INITIAL_API_TOKEN` — a random initial owner token, at least 32 characters.
   - `CDM_RAG_TRUSTED_PROXY_SECRET` — 32 random bytes encoded as 64 lowercase hex characters.
   - Optionally `CDM_RAG_EDGE_PORT` — a free host loopback port; default `19089`.

   Use randomly generated, distinct credentials; retain them in the owner's
   private secrets store. The import is initial only: changing these fields in
   `.env` does not update an already imported Keycloak realm or database user.
2. Verify `nvidia-ctk cdi list` contains `nvidia.com/gpu=all` and that Docker
   accepts CDI devices. This profile requests the CDI device directly and does
   not modify the daemon. Confirm disk and GPU availability before proceeding. Verify loopback ports
   19089–19095 are free on all interfaces. Host networking avoids the host
   firewall’s Docker bridge forwarding restrictions; it does not enforce
   isolation from other processes on the host.
3. Pull the images, start PostgreSQL and Ollama, and pull the following models
   into the **dedicated** Ollama volume: `qwen3-embedding:0.6b`, `qwen3:4b`,
   `qwen3.5:9b`, and `qwen2.5:0.5b`. Scope checks use `qwen3:4b`,
   matching the MacStudio configuration; the optional judge is disabled and
   uses `qwen2.5:0.5b` if enabled. Model downloads require network and space.
   Then start the remaining services. The app may restart until the embedding
   model is installed and its initial index is ready.
4. Stage an HTTPS edge configuration that serves only this hostname and forwards
   to the loopback port. Preserve the current config for rollback; validate the
   candidate before reload. The dedicated proxy overwrites forwarded hostname and
   scheme values. Keep direct host ports private and protect both the app and
   the Keycloak paths through the same TLS host.
5. Verify unauthenticated browser requests to `/`, `/RAG`, `/docs` and `/graph`
   are redirected to the independent Keycloak realm, and an unauthenticated
   `/api/v1/status` request is rejected with HTTP 401. Verify its admin
   and master-realm URLs are inaccessible externally. Then sign in with the
   owner account and verify a cited answer, graph, API access, GPU use, and
   index persistence. Do not claim completion based only on container health.

The realm import uses environment placeholders at its first start. Never publish
the imported realm, running-container configuration, database, or populated
`.env`. The identity database and app data are independent local volumes, not
off-host backups. Export and verify separate backups before relying on recovery.

## API access

Signed-in owners can manage application tokens at `/api-tokens`. New tokens are
shown once; stored records contain only hashes. Applications authenticate with
`Authorization: Bearer <token>`. Token management requires the owner session;
application tokens cannot create or revoke tokens. Revocation takes effect on the
next request. An initial owner token is seeded from the private `.env`; a revoked
initial token remains revoked after restart. Keep the `.env` and API token database
private, and protect backups of the database.

For terminal examples that ask questions, inspect sources and display progress,
see the [GPU API guide](../../docs/gpu-api.md).

## Checking an API token

Run the status check from a trusted machine with Python 3:

```sh
python3 scripts/check_api_token.py --env-file /path/to/.env
```

The file must be explicitly named and may contain `CDM_RAG_GPU_PUBLIC_HOST` and
`CDM_RAG_GPU_INITIAL_API_TOKEN`. To check a different application token, set
`CDM_RAG_API_TOKEN` in the file or environment. Environment variables override
file values; `--url https://example.com` overrides the host. Protect the file
and do not pass a token on the command line. The check sends one authenticated
request to `/api/v1/status`, prints no token, validates the HTTPS certificate,
and does not follow redirects. It does not submit a question or run a model.

Exit status is `0` for an accepted token, `1` for HTTP 401 or 403, `2` for a
configuration error, and `3` for a network, TLS, redirect, unexpected response,
or other server error. An accepted token alone does not prove that unauthenticated
requests are rejected. If Python cannot verify the server certificate, configure
its trusted CA store or `SSL_CERT_FILE`; do not disable certificate validation.
