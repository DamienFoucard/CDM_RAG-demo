# Restricted demo access

The independent GPU deployment can admit a separately provisioned Keycloak user
and issue a demo-scoped API token. Demo credentials are separate from the owner
credentials. Add the Keycloak user with a unique verified email distinct from the
configured owner email; do not grant administrative roles. The login proxy passes
the verified email and OIDC subject to the app. The app requires both the
configured owner email and immutable subject for owner access; any signed-in
non-owner receives demo access. Obtain the owner's subject from the existing
Keycloak user, set `CDM_RAG_OWNER_SUBJECT`, and verify it against an actual
owner authentication response before changing the running deployment. Do not
use a username or mutable email as the subject.

Remove the default composite realm role from the demo user if it grants
Keycloak's account-management roles, and verify effective roles after creation.
The gateway blocks Keycloak's account self-service paths and only forwards the
normal sign-in parameters to the OIDC authorization endpoint, excluding
application-initiated actions. Test the actual realm and login configuration
before issuing credentials. This gateway restriction also
removes public account self-service for the owner.

Set `CDM_RAG_DEMO_EXPIRES_AT` in the protected deployment environment to an
explicit ISO-8601 time with timezone (for example, a UTC `Z` timestamp). An
absent, invalid, or elapsed deadline disables all demo access while preserving
owner access. This app-enforced deadline does not disable the Keycloak account;
disable the account and revoke its sessions when the handover is complete.

An owner can create a **Demo** token on the API tokens page. Tokens are shown
once, stored as hashes in a separate table, expire after seven days, and can be
revoked there. Existing tokens retain full API access; create a new demo token for
the demonstration. Do not put credentials or tokens into a repository, URL,
application log, or shared issue. Distribute them through an appropriate secure
channel and revoke the token and disable the user when the demo is over.

Demo sessions and demo tokens have an explicit allowlist: schema questions,
status, read-only graph operations, and a small set of pages. Demo questions
use hybrid search with up to six results. They use the server's reranking
default and can turn reranking off for a request, but cannot turn it on when
the server default is off. The independent GPU profile enables it by default.
A bearer token is required for MCP. Demo clients cannot change model settings,
manage tokens, access evaluation data or saved runs, or execute benchmark jobs.
Demo status omits private paths and configuration. A full token retains API
access to settings and benchmarks, but cannot administer tokens; that requires
the owner session.

The independent deployment runs one app worker. It applies shared, process-local
demo limits of 120 API requests per minute, 12 REST questions per hour, and 36
MCP POST requests per hour, including protocol setup. One demo question or MCP
POST may be active at a time. These limits apply to all demo accounts together
and reset if the app restarts. They bound casual workload
but are not distributed quotas or a substitute for edge abuse protection; use a
shared rate limiter before running multiple workers or replicas. Queries still
share the same GPU and model lock with the owner and are stored in the app's
query log. Direct host-loopback access bypasses the public proxy and must remain
trusted. Test authorization and workload isolation before sharing access.

## Rollback

The original full-token table is unchanged, and old releases cannot validate
the separate demo tokens. However, older versions grant full access to any
authenticated browser session. Before restoring an older app or gateway, block
external access, disable the demo user, revoke its Keycloak sessions, rotate the
independent OAuth2-proxy cookie secret and restart that proxy. Verify an old demo
cookie is rejected before reopening the edge. Rotation invalidates all browser
sessions, so the owner must sign in again. Keep a consistent private backup of
the token database and deployment configuration before rollout or rollback.
