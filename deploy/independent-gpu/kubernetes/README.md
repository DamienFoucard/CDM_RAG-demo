# Flux-managed application on the independent GPU host

This Kustomize base lets Flux deploy the application while the independent
Compose stack supplies Keycloak, PostgreSQL, the login proxy, Nginx and Ollama.
The application binds to loopback on that same host. The host, cluster and
local processes are inside its trust boundary. An installation may need a
namespace where admission permits host networking.

The deployment artifact must replace the placeholder image with a published
image digest. Configure the installation's Flux Kustomization to supply the
namespace, intended node hostname, host loopback application port and Ollama
URL. The values in the base match the public Compose defaults. Supply an
image pull Secret when the registry requires authentication.

Provision the `cdm-rag-runtime` Secret with the same values as the independent
Compose installation for `CDM_RAG_OWNER_EMAIL`, `CDM_RAG_OWNER_SUBJECT`,
`CDM_RAG_PUBLIC_HOST` and `CDM_RAG_TRUSTED_PROXY_SECRET`. Preserve
`CDM_RAG_DEMO_EXPIRES_AT` if guest access is in use. Existing token hashes are
in the app data. `CDM_RAG_INITIAL_API_TOKEN` is optional and seeds a token when
provisioning a fresh instance. Keep runtime secrets outside the release
artifact and protect the GitOps copy through encryption.

Bind the `cdm-rag-data` claim to the existing app data with a local volume
that retains its contents. The app runs as UID and GID 10001 and needs write
access to the data directory. This directory holds the token registry, index,
model settings and query logs. It is separate from identity and Ollama data.
Schedule one app replica on the host with the volume. Updates use the
Kubernetes Recreate strategy to avoid overlapping writers.

For the initial switch, reconcile at zero replicas first and verify the source,
image access and volume binding. Drain active queries, stop the Compose `app`
service, take a protected backup of the app data, then reconcile one replica at
the existing loopback port. Verify deployment readiness, a public login redirect
and an authenticated API status response. To roll back during this initial
migration, set the Flux deployment to zero replicas and wait for the pod to
stop before starting the previous Compose app. The order protects both the
exclusive loopback port and the data from concurrent writers.
