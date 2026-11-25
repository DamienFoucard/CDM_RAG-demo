"""Self-contained FastAPI demonstration of retrieval-augmented CDM answers."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .index import SchemaIndex, tokens
from .benchmark import (BENCHMARK_LOG_PATH, available_chat_models, load_benchmark_questions,
                        benchmark_model_defaults, list_saved_benchmark_runs, load_saved_benchmark_run,
                        max_model_size_bytes, run_answer_benchmark)
from .retrieval_benchmark import (RETRIEVAL_BENCHMARK_LOG_PATH, available_embedding_models,
                                  append_retrieval_benchmark_run, load_catalog_object_dataset,
                                  load_contextual_retrieval_dataset, load_retrieval_dataset,
                                  list_saved_retrieval_runs,
                                  load_saved_retrieval_run, run_retrieval_benchmark)
from .query_retrieval_benchmark import reranker_options, run_query_retrieval_benchmark
from .adaptive_retrieval_benchmark import CHALLENGE_DATASET_PATH, load_adaptive_retrieval_dataset, run_adaptive_retrieval_benchmark
from .graph_retrieval_benchmark import (GRAPH_METHODS, load_relationship_datasets,
                                        run_relationship_retrieval_benchmark)
from .link_prediction_benchmark import (LINK_PREDICTION_METHODS,
                                        LINK_PREDICTION_METHOD_LABELS,
                                        link_prediction_graph_summary,
                                        run_link_prediction_benchmark)
from .graph_analysis import analyze_graph
from .observability import QUERY_LOG_PATH, QueryRecorder, new_query_id
from .ollama import ModelError, Ollama
from .rag import answer_question, generation_passage_limit
from .router import JevRouter
from .decision_router import LocalDecisionRouter
from .api_tokens import APIAuthMiddleware, TokenRegistry, run_query_worker
from .query_trace import TracedModels, active_trace
from .mcp_server import AppBackend, create_server


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "seed" / "catalog.json"
DATA_DIR = Path(os.getenv("CDM_RAG_DATA_DIR", str(ROOT / "data")))
MODEL_SETTINGS_PATH = DATA_DIR / "model_settings.json"
MODEL_ROLES = ("answer_model", "router_model", "decision_model", "scope_model", "judge_model")
logger = logging.getLogger(__name__)


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean value.")


def load_saved_model_roles() -> dict[str, str]:
    """Load API-wide local model choices, ignoring malformed or unknown settings."""
    try:
        payload = json.loads(MODEL_SETTINGS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning("Could not read saved model settings from %s", MODEL_SETTINGS_PATH)
        return {}
    if not isinstance(payload, dict):
        logger.warning("Saved model settings are not an object; using environment defaults")
        return {}
    return {role: value for role in MODEL_ROLES
            if isinstance((value := payload.get(role)), str) and value.strip()}


def save_model_roles(roles: dict[str, str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = MODEL_SETTINGS_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(roles, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, MODEL_SETTINGS_PATH)


def current_model_roles(models: Ollama) -> dict[str, str]:
    return {role: getattr(models, role) for role in MODEL_ROLES}


def active_router_backend(router: JevRouter | LocalDecisionRouter | None) -> str:
    if isinstance(router, LocalDecisionRouter):
        return "ollama-evidence" if router.use_evidence else "ollama-decision"
    return "jev" if router is not None else "ollama"


@asynccontextmanager
async def lifespan(app: FastAPI):
    models = Ollama(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                    os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
                    os.getenv("ROUTER_MODEL", "qwen3:4b"),
                    os.getenv("ANSWER_MODEL", "qwen3:4b"),
                    os.getenv("SCOPE_MODEL", "qwen3:4b"),
                    os.getenv("JUDGE_MODEL", "qwen3:4b"))
    models.decision_model = os.getenv("CDM_RAG_DECISION_MODEL", models.router_model)
    for role, model_name in load_saved_model_roles().items():
        setattr(models, role, model_name)
    judge_enabled = env_flag("CDM_RAG_JUDGE_ENABLED", False)
    router_backend = os.getenv("CDM_RAG_ROUTER_BACKEND", "ollama").strip().lower()
    if router_backend == "ollama":
        query_router = None
    elif router_backend in {"ollama-decision", "ollama-evidence"}:
        query_router = LocalDecisionRouter(models.decision_model, use_evidence=router_backend == "ollama-evidence")
    elif router_backend == "jev":
        try:
            minimum_confidence = float(os.getenv("CDM_RAG_ROUTER_MIN_CONFIDENCE", "0.70"))
        except ValueError as exc:
            raise RuntimeError("CDM_RAG_ROUTER_MIN_CONFIDENCE must be between 0 and 1.") from exc
        query_router = JevRouter(os.getenv("JEV_API_KEY", ""),
                                 os.getenv("JEV_MODEL", "jev-latest"),
                                 os.getenv("JEV_BASE_URL", "https://api.typesafe.ai"),
                                 minimum_confidence)
    else:
        raise RuntimeError("CDM_RAG_ROUTER_BACKEND must be 'ollama', 'ollama-decision', 'ollama-evidence' or 'jev'.")
    app.state.models = models
    app.state.judge_enabled = judge_enabled
    app.state.query_router = query_router
    app.state.router_backend = router_backend
    app.state.model_run_lock = threading.Lock()
    app.state.graph_analysis_lock = threading.Lock()
    app.state.benchmark_questions = load_benchmark_questions()
    app.state.retrieval_benchmark_dataset = load_retrieval_dataset()
    app.state.catalog_object_retrieval_dataset = load_catalog_object_dataset()
    app.state.contextual_retrieval_dataset = load_contextual_retrieval_dataset()
    if (app.state.catalog_object_retrieval_dataset["source_commit"]
            != app.state.retrieval_benchmark_dataset["source_commit"]):
        raise RuntimeError("The catalog-object retrieval labels target a different CDM source revision.")
    if app.state.contextual_retrieval_dataset["source_commit"] != app.state.retrieval_benchmark_dataset["source_commit"]:
        raise RuntimeError("The contextual retrieval labels target a different CDM source revision.")
    app.state.relationship_benchmark_dataset = load_relationship_datasets()
    app.state.index = await asyncio.to_thread(SchemaIndex, CATALOG, DATA_DIR, models)
    await asyncio.to_thread(models.unload_if_loaded, models.embedding_model)
    try:
        async with mcp_server.session_manager.run():
            yield
    finally:
        if isinstance(query_router, JevRouter):
            query_router.close()


app = FastAPI(title="CDM RAG", version="1.0.0", lifespan=lifespan,
              description="Answer questions about Microsoft CDM banking and common definitions using cited retrieval evidence.",
              docs_url=None, redoc_url=None)


API_AUTH_REQUIRED = env_flag("CDM_RAG_API_AUTH_REQUIRED", False)
API_PROXY_SECRET = os.getenv("CDM_RAG_TRUSTED_PROXY_SECRET", "")
if API_AUTH_REQUIRED and (len(API_PROXY_SECRET) != 64 or any(c not in "0123456789abcdef" for c in API_PROXY_SECRET)):
    raise RuntimeError("CDM_RAG_TRUSTED_PROXY_SECRET must be a 64-character lowercase hexadecimal secret.")
API_TOKEN_REGISTRY = TokenRegistry(DATA_DIR / "api_tokens.sqlite3", os.getenv("CDM_RAG_INITIAL_API_TOKEN")) if API_AUTH_REQUIRED else None
app.add_middleware(APIAuthMiddleware, required=API_AUTH_REQUIRED, registry=API_TOKEN_REGISTRY, proxy_secret=API_PROXY_SECRET,
                   owner_email=os.getenv("CDM_RAG_OWNER_EMAIL", ""),
                   owner_subject=os.getenv("CDM_RAG_OWNER_SUBJECT", ""),
                   demo_expires_at=os.getenv("CDM_RAG_DEMO_EXPIRES_AT", ""))
mcp_server = create_server(AppBackend(app), public_host=os.getenv("CDM_RAG_PUBLIC_HOST", ""))
app.mount('/api/v1/mcp', mcp_server.streamable_http_app())


class TokenCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=80)
    scope: Literal['full', 'demo'] = 'full'

    @field_validator("label")
    @classmethod
    def valid_label(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(c) < 32 for c in value):
            raise ValueError("Enter a label without control characters.")
        return value


def owner_origin(request: Request) -> None:
    expected = "https://" + os.getenv("CDM_RAG_PUBLIC_HOST", "")
    if request.headers.get("origin") != expected:
        raise HTTPException(status_code=403, detail="Owner session origin required.")


@app.get("/api-tokens", include_in_schema=False)
def api_token_page() -> FileResponse:
    if API_TOKEN_REGISTRY is None:
        raise HTTPException(status_code=404)
    return FileResponse(ROOT / "web" / "api-tokens.html", headers={"Cache-Control": "no-store"})


@app.get("/api/v1/api-tokens")
def list_api_tokens(response: Response) -> dict:
    if API_TOKEN_REGISTRY is None:
        raise HTTPException(status_code=404)
    response.headers["Cache-Control"] = "no-store"
    return {"tokens": API_TOKEN_REGISTRY.list()}


@app.post("/api/v1/api-tokens", status_code=201)
def create_api_token(body: TokenCreate, request: Request, response: Response) -> dict:
    if API_TOKEN_REGISTRY is None:
        raise HTTPException(status_code=404)
    owner_origin(request)
    response.headers["Cache-Control"] = "no-store"
    return API_TOKEN_REGISTRY.create(body.label, body.scope)


@app.delete("/api/v1/api-tokens/{token_id}", status_code=204)
def revoke_api_token(token_id: UUID, request: Request) -> Response:
    if API_TOKEN_REGISTRY is None:
        raise HTTPException(status_code=404)
    owner_origin(request)
    if not API_TOKEN_REGISTRY.revoke(str(token_id)):
        raise HTTPException(status_code=404, detail="Active token not found.")
    return Response(status_code=204, headers={"Cache-Control": "no-store"})


DEFAULT_RERANK_ENABLED = env_flag("CDM_RAG_RERANK_ENABLED", False)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["hybrid", "semantic", "lexical"] = "hybrid"
    rerank: bool = DEFAULT_RERANK_ENABLED
    limit: int = Field(default=6, ge=1, le=12)

    @field_validator("question")
    @classmethod
    def nonempty(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("enter a question")
        return value


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[str] = Field(min_length=1, max_length=6)
    question_ids: list[str] = Field(min_length=1, max_length=8)
    judge_model: str | None = Field(default=None, min_length=1, max_length=200)


class RetrievalBenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[str] = Field(default_factory=list, max_length=6)
    question_ids: list[str] = Field(default_factory=list, max_length=42)
    split: Literal["development", "test", "complex", "challenge"]
    track: Literal["passages", "catalog_objects", "relationships", "query_processing", "reranking", "link_prediction", "adaptive_retrieval"] = "passages"
    graph_methods: list[Literal["text_hybrid", "gcn_1", "gcn_2", "gcn_3", "transe", "transh", "boxe"]] = Field(
        default_factory=lambda: list(GRAPH_METHODS), max_length=7)
    query_methods: list[Literal["raw", "planner", "rewrite"]] = Field(default_factory=list, max_length=3)
    query_method: Literal["raw", "planner", "rewrite"] | None = None
    query_model: str | None = Field(default=None, min_length=1, max_length=200)
    query_models: list[str] = Field(default_factory=list, max_length=6)
    question_set: Literal["entities", "catalog_objects", "contextual"] = "entities"
    adaptive_question_set: Literal["v2", "reformulation_challenge_candidate"] = "v2"
    rerankers: list[Literal["ollama_llm", "bce", "bge"]] = Field(default_factory=list, max_length=3)

class ModelSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer_model: str | None = Field(default=None, min_length=1, max_length=200)
    router_model: str | None = Field(default=None, min_length=1, max_length=200)
    decision_model: str | None = Field(default=None, min_length=1, max_length=200)
    scope_model: str | None = Field(default=None, min_length=1, max_length=200)
    judge_model: str | None = Field(default=None, min_length=1, max_length=200)


@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    return RedirectResponse("/home", status_code=307)


@app.get("/home", include_in_schema=False)
def project_home_page() -> FileResponse:
    return FileResponse(ROOT / "web" / "home.html")


@app.get("/RAG", include_in_schema=False)
def rag_page() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/docs", include_in_schema=False)
def documentation_page() -> FileResponse:
    return FileResponse(ROOT / "web" / "documentation.html")


@app.get("/docs/design", include_in_schema=False)
def design_documentation_page() -> FileResponse:
    return FileResponse(ROOT / "web" / "design.html")


@app.get("/docs/mcp", include_in_schema=False)
def mcp_documentation_page() -> FileResponse:
    return FileResponse(ROOT / "web" / "mcp.html")


@app.get("/docs/images/cdm-rag-readme-tour.svg", include_in_schema=False)
def documentation_diagram() -> FileResponse:
    return FileResponse(ROOT / "docs" / "images" / "cdm-rag-readme-tour.svg",
                        media_type="image/svg+xml")


@app.get("/docs/api", include_in_schema=False)
def api_documentation_page() -> HTMLResponse:
    page = get_swagger_ui_html(openapi_url=app.openapi_url, title="CDM RAG · API reference",
                               swagger_ui_parameters={"deepLinking": True})
    html = page.body.decode("utf-8")
    html = html.replace("</head>", """<style>
      .docs-nav{display:flex;gap:20px;align-items:center;padding:14px 22px;background:#f4f7f4;border-bottom:1px solid #d9e5dc;font:700 14px/1.4 system-ui,sans-serif}
      .docs-nav a{color:#1d7653;text-decoration:none}.docs-nav a:hover{text-decoration:underline}
    </style></head>""", 1)
    html = html.replace("<body>", """<body><nav class="docs-nav" aria-label="Documentation navigation">
      <a href="/home">Project home</a><a href="/RAG">Ask a question</a>
      <a href="/docs">Documentation home</a><a href="/docs/design">Design overview</a><a href="/docs/mcp">MCP integration</a>
    </nav>""", 1)
    return HTMLResponse(html)


@app.get("/benchmarks", include_in_schema=False)
def benchmarks_page() -> FileResponse:
    return FileResponse(ROOT / "web" / "benchmarks.html")


@app.get("/benchmark", include_in_schema=False)
def benchmark_page() -> FileResponse:
    return FileResponse(ROOT / "web" / "benchmark.html")


@app.get("/routing-benchmark", include_in_schema=False)
def routing_benchmark_page() -> FileResponse:
    return FileResponse(ROOT / "web" / "routing-benchmark.html")


@app.get("/evaluation/routing-decision-v1.json", include_in_schema=False)
def routing_decision_dataset() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "routing-decision-v1.json", media_type="application/json")


@app.get("/evaluation/routing-scope-binary-v1.json", include_in_schema=False)
def routing_scope_binary_dataset() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "routing-scope-binary-v1.json", media_type="application/json")


@app.get("/evaluation/scope-questions-v1.json", include_in_schema=False)
def routing_scope_source_dataset() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "scope-questions-v1.json", media_type="application/json")


@app.get("/evaluation/results/routing-scope-binary-snapshot.json", include_in_schema=False)
def routing_scope_binary_snapshot() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "results" / "routing-scope-binary-snapshot.json",
                        media_type="application/json")


@app.get("/evaluation/results/routing-evidence-aware-snapshot.json", include_in_schema=False)
def routing_evidence_aware_snapshot() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "results" / "routing-evidence-aware-snapshot.json",
                        media_type="application/json")


@app.get("/evaluation/results/routing-scope-evidence-aware-snapshot.json", include_in_schema=False)
def routing_scope_evidence_aware_snapshot() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "results" / "routing-scope-evidence-aware-snapshot.json",
                        media_type="application/json")


@app.get("/evaluation/results/routing-cosine-snapshot.json", include_in_schema=False)
def routing_cosine_snapshot() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "results" / "routing-cosine-snapshot.json",
                        media_type="application/json")


@app.get("/evaluation/results/routing-benchmark-snapshot.json", include_in_schema=False)
def routing_benchmark_snapshot() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "results" / "routing-benchmark-snapshot.json",
                        media_type="application/json")


@app.get("/evaluation/results/generation-output-snapshot.json", include_in_schema=False)
def generation_output_snapshot() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "results" / "generation-output-snapshot.json",
                        media_type="application/json")


@app.get("/evaluation/results/retrieval-qwen35-27b-output-snapshot.json", include_in_schema=False)
def qwen35_27b_retrieval_output_snapshot() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "results" / "retrieval-qwen35-27b-output-snapshot.json",
                        media_type="application/json")


@app.get("/evaluation/results/retrieval-historical-output-snapshot.json", include_in_schema=False)
def retrieval_historical_output_snapshot() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "results" / "retrieval-historical-output-snapshot.json",
                        media_type="application/json")


@app.get("/evaluation/results/macstudio-2026-11-24-fixed-evidence-blind-review.md", include_in_schema=False)
def fixed_evidence_blind_review_report() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "results" / "macstudio-2026-11-24-fixed-evidence-blind-review.md",
                        media_type="text/markdown")


@app.get("/evaluation/results/macstudio-2026-11-24-context-position-test.md", include_in_schema=False)
def context_position_report() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "results" / "macstudio-2026-11-24-context-position-test.md",
                        media_type="text/markdown")


@app.get("/evaluation/results/macstudio-2026-11-24-new-reasoning-models.md", include_in_schema=False)
def new_reasoning_models_report() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "results" / "macstudio-2026-11-24-new-reasoning-models.md",
                        media_type="text/markdown")


@app.get("/evaluation/results/macstudio-qwen35-27b-thinking-retrieval.md", include_in_schema=False)
def qwen35_27b_thinking_retrieval_report() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "results" / "macstudio-qwen35-27b-thinking-retrieval.md",
                        media_type="text/markdown")


@app.get("/evaluation/results/macstudio-qwen35-27b-thinking-answers.md", include_in_schema=False)
def qwen35_27b_thinking_answers_report() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "results" / "macstudio-qwen35-27b-thinking-answers.md",
                        media_type="text/markdown")


@app.get("/evaluation/results/macstudio-2026-11-24-reformulation-followup.md", include_in_schema=False)
def reformulation_followup_report() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "results" / "macstudio-2026-11-24-reformulation-followup.md",
                        media_type="text/markdown")


@app.get("/evaluation/results/macstudio-graph-expansion-replay.md", include_in_schema=False)
def graph_expansion_replay_report() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "results" / "macstudio-graph-expansion-replay.md",
                        media_type="text/markdown")


@app.get("/retrieval-benchmark", include_in_schema=False)
def retrieval_benchmark_page() -> FileResponse:
    return FileResponse(ROOT / "web" / "retrieval-benchmark.html")


@app.get("/embedding-benchmark", include_in_schema=False)
def embedding_benchmark_page() -> FileResponse:
    return FileResponse(ROOT / "web" / "embedding-benchmark.html")


@app.get("/graph", include_in_schema=False)
def graph_explorer_page() -> FileResponse:
    return FileResponse(ROOT / "web" / "graph.html")


@app.get("/api/v1/graph/overview", include_in_schema=False)
def graph_overview() -> dict:
    graph = app.state.index.graph
    semantics = graph.semantic_layers()
    banking_nodes = [
        entity_id for entity_id, entity in graph.entities.items()
        if entity.get("model") == "RetailBankingCoreDataModel" and graph.adjacency[entity_id]
    ]
    candidates = banking_nodes or list(graph.entities)
    # Start with a useful, readable finance neighborhood instead of an
    # administrative hub with more than a hundred audit and ownership links.
    initial_entity_id = min(
        candidates,
        key=lambda entity_id: (abs(min(len(graph.adjacency[entity_id]), 12) - 8),
                               -len(graph.adjacency[entity_id]),
                               graph.entities[entity_id]["name"].casefold(), entity_id),
    )
    return {
        "entity_count": len(graph.entities),
        "relationship_count": len(graph.edges),
        "trait_count": len(semantics["traits"]["nodes"]),
        "manifest_count": sum(
            not item.get("declaration_only", False)
            for item in semantics["manifests"]["nodes"]
        ),
        "unresolved_reference_count": len(graph.unresolved),
        "source_revision": graph.revision,
        "initial_entity_id": initial_entity_id,
        "initial_entity_name": graph.entities[initial_entity_id]["name"],
        "coverage": "loaded_catalog_subset",
    }


@app.get("/api/v1/graph/all", include_in_schema=False)
def graph_all() -> dict:
    """Return the full loaded CDM schema graph for the infinite-hops explorer."""
    graph = app.state.index.graph
    entity_details = {
        entity_id: {
            "attributes": [
                {
                    "name": item.get("name", ""),
                    "display_name": item.get("display_name", ""),
                    "data_type": item.get("data_type", ""),
                    "description": item.get("description", ""),
                    "path": item.get("path", ""),
                    "applied_traits": item.get("applied_traits", []),
                    "data_type_traits": item.get("data_type_traits", []),
                }
                for item in entity.get("attributes", [])
            ],
            "applied_traits": entity.get("applied_traits", []),
            "extends_entity": entity.get("extends_entity"),
        }
        for entity_id, entity in graph.entities.items()
    }
    return {
        "source": graph.source,
        "source_revision": graph.revision,
        "graph_revision": graph.projection_revision,
        "coverage": "loaded_catalog_subset",
        "truncated": False,
        "depth": "infinite",
        "entity_count": len(graph.entities),
        "relationship_count": len(graph.edges),
        "nodes": [graph._node(entity_id) for entity_id in graph.entities],
        "edges": list(graph.edges.values()),
        "entity_details": entity_details,
        "semantics": graph.semantic_layers(),
        "history_saved": False,
    }


@app.post("/api/v1/graph/analysis", include_in_schema=False)
async def graph_analysis() -> dict:
    """Compute a disk-cached embedding layout and local-model entity categories."""
    def run_analysis() -> dict:
        with app.state.graph_analysis_lock, app.state.model_run_lock:
            return analyze_graph(app.state.index, app.state.models, DATA_DIR)

    try:
        return await asyncio.to_thread(run_analysis)
    except (RuntimeError, OSError, ValueError) as exc:
        logger.exception("Could not analyze the full CDM graph")
        raise HTTPException(status_code=503, detail="The graph layout and categories could not be prepared.") from exc


@app.get("/api/v1/graph/search", include_in_schema=False)
def graph_search(q: str = Query(min_length=1, max_length=100),
                 limit: int = Query(default=40, ge=1, le=80)) -> dict:
    """Find entity, attribute, and relationship locations without recording searches."""
    graph = app.state.index.graph
    normalized_query = " ".join(q.split())
    query_terms = tokens(normalized_query)
    if not query_terms:
        return {"query": normalized_query, "hits": [], "source_revision": graph.revision}

    hits: list[dict] = []

    def add_hit(*, hit_id: str, kind: str, title: str, subtitle: str,
                entity: dict, searchable: list[str], snippet: str,
                member_name: str | None = None, edge_id: str | None = None,
                location_path: str | None = None) -> None:
        fields = [title, *searchable]
        field_tokens = set(tokens(" ".join(item for item in fields if item)))
        if not all(term in field_tokens for term in query_terms):
            return
        title_tokens = set(tokens(title))
        title_matches = sum(term in title_tokens for term in query_terms)
        score = title_matches * 10 + sum(
            term in set(tokens(" ".join(searchable))) for term in query_terms
        ) * 2
        if " ".join(tokens(title)) == " ".join(query_terms):
            score += 100
        hits.append({
            "id": hit_id,
            "kind": kind,
            "title": title,
            "subtitle": subtitle,
            "snippet": snippet,
            "entity_id": entity["id"],
            "entity_name": entity["name"],
            "model": entity["model"],
            "path": location_path or entity.get("path", ""),
            "member_name": member_name,
            "edge_id": edge_id,
            "score": score,
        })

    for entity in graph.entities.values():
        add_hit(
            hit_id=f"entity:{entity['id']}", kind="entity", title=entity["name"],
            subtitle=f"Entity · {entity['model']}", entity=entity,
            searchable=[entity.get("model", ""), entity.get("description", "")],
            snippet=entity.get("description", "") or "Entity definition",
        )
        for attribute in entity.get("attributes", []):
            attribute_name = attribute.get("name", "")
            add_hit(
                hit_id=f"attribute:{entity['id']}:{attribute_name}", kind="attribute",
                title=attribute_name, subtitle=f"Attribute · {entity['name']} · {entity['model']}",
                entity=entity,
                searchable=[entity["name"], entity["model"], attribute.get("display_name", ""),
                            attribute.get("data_type", ""), attribute.get("description", "")],
                snippet=attribute.get("description", "") or attribute.get("data_type", "CDM attribute"),
                member_name=attribute_name, location_path=attribute.get("path"),
            )

    for edge in graph.edges.values():
        source = graph.entities[edge["source_id"]]
        target = graph.entities[edge["target_id"]]
        title = f"{source['name']}.{edge['attribute']} → {target['name']}"
        add_hit(
            hit_id=f"relationship:{edge['id']}", kind="relationship", title=title,
            subtitle=f"Relationship · {source['model']}", entity=source,
            searchable=[source["name"], source["model"], target["name"], target["model"],
                        edge.get("attribute", ""), edge.get("description", "")],
            snippet=edge.get("description", "") or f"{source['name']} links to {target['name']} through {edge['attribute']}.",
            member_name=edge.get("attribute"), edge_id=edge["id"], location_path=edge.get("path"),
        )

    hits.sort(key=lambda hit: (-hit["score"],
                               {"entity": 0, "attribute": 1, "relationship": 2}[hit["kind"]],
                               hit["title"].casefold(), hit["entity_id"]))
    for hit in hits:
        hit.pop("score", None)
    return {"query": normalized_query, "hits": hits[:limit],
            "total_hits": len(hits), "source_revision": graph.revision,
            "history_saved": False}


@app.get("/api/v1/graph/neighborhood", include_in_schema=False)
def graph_neighborhood(entity_id: str = Query(min_length=1, max_length=300),
                       depth: int = Query(default=1, ge=1, le=2)) -> dict:
    graph = app.state.index.graph
    entity = graph.entities.get(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found in the loaded CDM catalog.")
    neighborhood = graph.neighborhood([entity_id], depth=depth, max_nodes=30,
                                      max_edges=60, max_context_bytes=100_000)
    direct_relationships = []
    for edge in graph.adjacency[entity_id]:
        outgoing = edge["source_id"] == entity_id
        other_id = edge["target_id"] if outgoing else edge["source_id"]
        other = graph.entities[other_id]
        direct_relationships.append({
            **edge,
            "direction": "outgoing" if outgoing else "incoming",
            "other_entity_id": other_id,
            "other_name": other["name"],
            "other_model": other["model"],
        })
    return {
        **neighborhood,
        "focus_entity": {
            "id": entity["id"], "name": entity["name"], "model": entity["model"],
            "description": entity.get("description", ""), "path": entity.get("path", ""),
            "source_uri": entity.get("source_uri", ""),
            "attributes": [{"name": item.get("name", ""), "display_name": item.get("display_name", ""),
                            "data_type": item.get("data_type", ""),
                            "description": item.get("description", ""),
                            "path": item.get("path", "")}
                           for item in entity.get("attributes", [])],
        },
        "direct_relationships": direct_relationships,
    }


@app.get("/evaluation/cdm-schema-evaluation-v2.json", include_in_schema=False)
def evaluation_dataset() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "cdm-schema-evaluation-v2.json",
                        media_type="application/json")


@app.get("/evaluation/answer-benchmark-v1.json", include_in_schema=False)
def answer_evaluation_dataset() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "answer-benchmark-v1.json",
                        media_type="application/json")


@app.get("/evaluation/catalog-object-retrieval-v1.json", include_in_schema=False)
def catalog_object_evaluation_dataset() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "catalog-object-retrieval-v1.json",
                        media_type="application/json")


@app.get("/evaluation/relationship-questions-v1.json", include_in_schema=False)
def relationship_evaluation_dataset() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "relationship-questions-v1.json",
                        media_type="application/json")


@app.get("/evaluation/relationship-challenge-v1.json", include_in_schema=False)
def relationship_challenge_dataset() -> FileResponse:
    return FileResponse(ROOT / "evaluation" / "relationship-challenge-v1.json",
                        media_type="application/json")


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/api/v1/status")
def status(request: Request) -> dict:
    index = app.state.index
    if getattr(request.state, 'demo', False):
        return {"demo_access": True, "entity_count": len(index.catalog["entities"]), "passage_count": len(index.chunks),
                "source_commit": index.catalog["source_commit"]}
    query_router = app.state.query_router
    return {"entity_count": len(index.catalog["entities"]), "passage_count": len(index.chunks),
            "source_commit": index.catalog["source_commit"], "index_revision": index.index_revision,
            "embedding_model": app.state.models.embedding_model,
            "router_model": query_router.model if query_router is not None else app.state.models.router_model,
            "local_router_model": app.state.models.router_model,
            "decision_model": app.state.models.decision_model,
            "answer_model": app.state.models.answer_model,
            "answer_runtime": app.state.models.runtime_info(app.state.models.answer_model),
            "scope_model": app.state.models.scope_model,
            "judge_enabled": app.state.judge_enabled,
            "judge_model": app.state.models.judge_model,
            "router_backend": active_router_backend(query_router),
            "reranking_enabled_by_default": DEFAULT_RERANK_ENABLED,
            "reranker_model": app.state.models.answer_model,
            "generation_passage_limit": generation_passage_limit(),
            "query_embedding_keep_alive": app.state.models.query_embedding_keep_alive,
            "query_log_path": str(QUERY_LOG_PATH),
            "benchmark_log_path": str(BENCHMARK_LOG_PATH),
            "first_token_timeout_seconds": app.state.models.first_token_timeout,
            "stream_idle_timeout_seconds": app.state.models.stream_idle_timeout}


def _model_settings_payload(installed_models: list[dict]) -> dict:
    models = app.state.models
    return {
        "models": installed_models,
        "selected": current_model_roles(models),
        "embedding_model": models.embedding_model,
        "embedding_model_note": "Index-bound; changing it requires rebuilding the schema index.",
        "router_backend": active_router_backend(app.state.query_router),
        "judge_enabled": app.state.judge_enabled,
        "max_model_size_gb": round(max_model_size_bytes() / (1024 ** 3), 2),
        "settings_path": str(MODEL_SETTINGS_PATH),
    }


@app.get("/api/v1/settings/models")
def get_model_settings() -> dict:
    """Return process-wide model roles and locally installed choices."""
    try:
        installed_models = available_chat_models(app.state.models.base_url)
    except ModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _model_settings_payload(installed_models)


@app.put("/api/v1/settings/models")
def update_model_settings(body: ModelSettingsUpdate) -> dict:
    """Change local model roles for this API process and persist them on disk."""
    requested = body.model_dump(exclude_unset=True, exclude_none=True)
    if not requested:
        raise HTTPException(status_code=400, detail="Select at least one model role to update.")
    try:
        installed_models = available_chat_models(app.state.models.base_url)
    except ModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    installed_names = {item["name"] for item in installed_models}

    with app.state.model_run_lock:
        models = app.state.models
        current = current_model_roles(models)
        for role, model_name in requested.items():
            if model_name != current[role] and model_name not in installed_names:
                raise HTTPException(status_code=422,
                                    detail=(f"{model_name!r} is not an installed local chat model within "
                                            "the configured model size cap."))
        updated = {**current, **requested}
        try:
            save_model_roles(updated)
        except OSError as exc:
            logger.exception("Could not persist API-wide model settings")
            raise HTTPException(status_code=500, detail="Model choices could not be saved to disk.") from exc

        obsolete_models = sorted(set(current.values()) - set(updated.values()) - {models.embedding_model})
        unload_warnings = []
        for model_name in obsolete_models:
            try:
                models.unload_if_loaded(model_name)
            except ModelError:
                logger.warning("Changed model %s remains loaded in Ollama", model_name)
                unload_warnings.append(model_name)
        for role, model_name in updated.items():
            setattr(models, role, model_name)
        if isinstance(app.state.query_router, LocalDecisionRouter):
            app.state.query_router = LocalDecisionRouter(models.decision_model,
                                                         use_evidence=app.state.query_router.use_evidence)

    if unload_warnings:
        logger.warning("Updated models but could not unload old models: %s", ", ".join(unload_warnings))
    return {
        **_model_settings_payload(installed_models),
        "message": "Saved. New API requests will use the selected models.",
        "unload_warnings": unload_warnings,
    }


def _run_logged_query(body: Question, query_id: str, *,
                      on_stage: Callable[[str], None] | None = None,
                      on_token: Callable[[str], None] | None = None,
                      on_model: Callable[[dict[str, str]], None] | None = None,
                      request_started_at: float | None = None) -> dict:
    recorder: QueryRecorder | None = None
    started_at = request_started_at if request_started_at is not None else time.monotonic()
    if on_stage:
        on_stage("waiting_for_model")
    try:
        with app.state.model_run_lock:
            queue_ms = round((time.monotonic() - started_at) * 1000)
            recorder = QueryRecorder(
                query_id, body.question, mode=body.mode, rerank=body.rerank, limit=body.limit,
                models={"embedding": app.state.models.embedding_model,
                        "router": (app.state.query_router.model if app.state.query_router
                                   else app.state.models.router_model),
                        "query": app.state.models.router_model,
                        "scope": app.state.models.scope_model,
                        "judge": app.state.models.judge_model if app.state.judge_enabled else "disabled",
                        "answer": app.state.models.answer_model},
            )
            recorder.start()
            traced = TracedModels(app.state.models, lambda: recorder.current_stage, on_model)

            def stage(name: str) -> None:
                recorder.stage(name)
                if on_stage:
                    on_stage(name)
            trace_token = active_trace.set(traced)
            try:
                result = answer_question(
                    body.question, app.state.index, traced,
                    rerank_enabled=body.rerank, limit=body.limit, mode=body.mode,
                    router=app.state.query_router, judge_enabled=app.state.judge_enabled,
                    progress=stage, on_token=on_token, on_generation_input=recorder.generation_input,
                    before_generation=lambda: app.state.models.unload_other_models(app.state.models.answer_model),
                )
            finally:
                active_trace.reset(trace_token)
    except Exception as exc:
        if recorder is not None:
            recorder.fail(exc)
        raise
    assert recorder is not None
    recorder.complete(result)
    result["query_id"] = query_id
    calls = list(traced.calls)
    retrieval = result.get("retrieval", {})
    if retrieval.get("router_backend") == "ollama" and retrieval.get("query_generated"):
        for call in calls:
            if call["stage"] == "planning_search" and call["role"] == "routing":
                call["role"] = "routing_and_query"
    if retrieval.get("router_backend") == "jev":
        calls.append({"role": "routing", "model": retrieval.get("router_model", ""),
                      "stage": "planning_search", "outcome": "attempted"})
    result["processing"] = {
        "queue_ms": queue_ms,
        "elapsed_ms": round((time.monotonic() - started_at) * 1000),
        "stage_timings": list(recorder.stage_timings),
        "configured_models": recorder.models,
        "model_calls": calls,
        "answer_method": retrieval.get("generator") or "no_generation",
    }
    return result


def check_demo_query_options(request: Request, body: Question) -> None:
    if getattr(request.state, 'demo', False) and (
        (body.rerank and not DEFAULT_RERANK_ENABLED) or body.limit > 6 or body.mode != 'hybrid'
    ):
        raise HTTPException(status_code=403, detail='Demo questions use the default search settings.')


@app.post("/api/v1/query")
async def query(body: Question, response: Response, request: Request) -> dict:
    check_demo_query_options(request, body)
    query_id = new_query_id()
    request_started_at = time.monotonic()
    response.headers["X-Query-ID"] = query_id
    try:
        return await run_query_worker(_run_logged_query, body, query_id,
                                       request_started_at=request_started_at)
    except ModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"X-Query-ID": query_id}) from exc
    except Exception as exc:
        logger.exception("CDM query failed query_id=%s", query_id)
        raise HTTPException(status_code=500, detail="The query could not be completed.",
                             headers={"X-Query-ID": query_id}) from exc


@app.post("/api/v1/query/stream")
async def query_stream(body: Question, request: Request) -> StreamingResponse:
    check_demo_query_options(request, body)
    query_id = new_query_id()
    stream_started = time.monotonic()
    async def events():
        queue: asyncio.Queue[dict] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def progress(stage: str) -> None:
            if stage == "waiting_for_model":
                return  # Emitted before the worker starts, including executor wait.
            loop.call_soon_threadsafe(queue.put_nowait,
                                      {"type": "status", "stage": stage, "query_id": query_id,
                                       "elapsed_ms": round((time.monotonic() - stream_started) * 1000)})

        def token(value: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait,
                                      {"type": "token", "token": value, "query_id": query_id,
                                       "provisional": True})

        def model_call(details: dict[str, str]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait,
                                      {"type": "model", **details, "query_id": query_id,
                                       "elapsed_ms": round((time.monotonic() - stream_started) * 1000)})

        async def run_query() -> None:
            try:
                result = await run_query_worker(
                    _run_logged_query, body, query_id, on_stage=progress, on_token=token, on_model=model_call,
                    request_started_at=stream_started,
                )
            except ModelError as exc:
                await queue.put({"type": "error", "detail": str(exc), "query_id": query_id})
            except Exception:
                logger.exception("Streaming CDM query failed query_id=%s", query_id)
                await queue.put({"type": "error", "detail": "The query could not be completed.",
                                 "query_id": query_id})
            else:
                await asyncio.sleep(0)
                await queue.put({"type": "result", "result": result})

        worker = asyncio.create_task(run_query())
        try:
            yield f"data: {json.dumps({'type': 'status', 'stage': 'waiting_for_model', 'query_id': query_id, 'elapsed_ms': 0}, separators=(',', ':'))}\n\n"
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(item, separators=(',', ':'))}\n\n"
                if item["type"] in {"result", "error"}:
                    break
            await worker
        except asyncio.CancelledError:
            worker.cancel()
            raise

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Query-ID": query_id},
    )


@app.get("/api/v1/benchmark/options")
def benchmark_options() -> dict:
    try:
        models = available_chat_models(app.state.models.base_url)
    except ModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    dataset = app.state.benchmark_questions
    model_defaults = benchmark_model_defaults(models, app.state.models.base_url)
    return {
        "dataset_id": dataset["dataset_id"],
        "source_dataset": dataset["source_dataset"],
        "source_commit": dataset["source_commit"],
        "provenance": dataset["provenance"],
        "embedding_model": app.state.models.embedding_model,
        "configured_judge_model": app.state.models.judge_model,
        **model_defaults,
        "model_max_bytes": max_model_size_bytes(),
        "model_max_gb": round(max_model_size_bytes() / (1024 ** 3), 2),
        "models": models,
        "questions": [{
            "id": item["id"], "question": item["question"], "complexity": item["complexity"],
            "split": item["split"], "expected_entity_count": len(item["expected_entity_ids"]),
            "source_question_id": item.get("source_question_id"),
        } for item in dataset["questions"]],
        "log_path": str(BENCHMARK_LOG_PATH),
    }


@app.get("/api/v1/benchmark/runs")
def benchmark_runs() -> dict:
    try:
        runs = list_saved_benchmark_runs()
    except ModelError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"runs": runs, "log_path": str(BENCHMARK_LOG_PATH)}


@app.get("/api/v1/benchmark/runs/{run_id}")
def saved_benchmark_run(run_id: UUID) -> dict:
    try:
        result = load_saved_benchmark_run(str(run_id))
    except ModelError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="The saved benchmark run was not found.")
    return result


@app.post("/api/v1/benchmark/stream")
async def benchmark_stream(body: BenchmarkRequest) -> StreamingResponse:
    run_id = new_query_id()

    async def events():
        queue: asyncio.Queue[dict] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(item: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        async def run() -> None:
            try:
                await asyncio.to_thread(
                    run_answer_benchmark, app.state.index, app.state.models,
                    body.models, body.question_ids, app.state.benchmark_questions,
                    run_id, app.state.model_run_lock, emit, body.judge_model,
                )
            except Exception as exc:
                logger.exception("Answer benchmark failed run_id=%s", run_id)
                await queue.put({"type": "benchmark_error", "run_id": run_id,
                                 "detail": str(exc) if isinstance(exc, (ValueError, ModelError))
                                 else "The benchmark could not be completed."})

        worker = asyncio.create_task(run())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(item, separators=(',', ':'))}\n\n"
                if item["type"] in {"benchmark_completed", "benchmark_error"}:
                    break
            await worker
        except asyncio.CancelledError:
            raise

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Benchmark-ID": run_id},
    )


@app.get("/api/v1/retrieval-benchmark/options")
def retrieval_benchmark_options() -> dict:
    try:
        models = available_embedding_models(app.state.models.base_url)
    except ModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    dataset = app.state.retrieval_benchmark_dataset
    object_dataset = app.state.catalog_object_retrieval_dataset
    contextual_dataset = app.state.contextual_retrieval_dataset
    adaptive_dataset = load_adaptive_retrieval_dataset()
    challenge_dataset = load_adaptive_retrieval_dataset(CHALLENGE_DATASET_PATH)
    relationship_dataset = app.state.relationship_benchmark_dataset
    link_graph = link_prediction_graph_summary(CATALOG)
    link_prediction_best = None
    try:
        for run in list_saved_retrieval_runs():
            if (run.get("track") != "link_prediction" or run.get("split") != "development"
                    or run.get("source_commit") != link_graph["source_commit"]):
                continue
            candidates = [item for item in run.get("summaries", [])
                          if isinstance(item.get("metrics"), dict)
                          and isinstance(item["metrics"].get("mrr"), (int, float))]
            if candidates:
                winner = max(candidates, key=lambda item: item["metrics"]["mrr"])
                candidate = {
                    "run_id": run["run_id"], "started_at": run.get("started_at"),
                    "method": winner["method"], "label": winner.get("label"),
                    "feature_model": winner.get("feature_model"),
                    "mrr": winner["metrics"]["mrr"],
                }
                if (link_prediction_best is None
                        or candidate["mrr"] > link_prediction_best["mrr"]):
                    link_prediction_best = candidate
    except ModelError:
        link_prediction_best = None
    return {
        "dataset_id": dataset["dataset_id"],
        "base_dataset_id": dataset.get("base_dataset_id"),
        "supplemental_dataset_id": dataset.get("supplemental_dataset_id"),
        "dataset_status": dataset["status"],
        "dataset_provenance": dataset.get("provenance"),
        "supplemental_provenance": dataset.get("supplemental_provenance"),
        "source_commit": dataset["source_commit"],
        "model_max_bytes": max_model_size_bytes(),
        "model_max_gb": round(max_model_size_bytes() / (1024 ** 3), 2),
        "models": models,
        "query_models": available_chat_models(app.state.models.base_url),
        "default_query_model": app.state.models.router_model,
        "llm_reranker_model": app.state.models.answer_model,
        "reranker_methods": reranker_options(),
        "questions": [{
            "id": item["id"], "question": item["question"], "split": item["split"],
            "category": item.get("category"), "positive_entity_ids": item["positive_entity_ids"],
        } for item in dataset["questions"]],
        "object_dataset_id": object_dataset["dataset_id"],
        "object_dataset_provenance": object_dataset.get("provenance"),
        "object_questions": [{
            "id": item["id"], "question": item["question"], "split": item["split"],
            "category": item.get("category"), "object_type": item["object_type"],
            "expected_source_type": item["expected_source_type"],
            "expected_kind": item["expected_kind"],
            "positive_passage_ids": item["positive_passage_ids"],
        } for item in object_dataset["questions"]],
        "contextual_dataset_id": contextual_dataset["dataset_id"],
        "contextual_dataset_provenance": contextual_dataset.get("provenance"),
        "contextual_questions": [{
            "id": item["id"], "question": item["question"],
            "context_messages": item["context_messages"], "split": item["split"],
            "category": item["category"], "object_type": item.get("object_type"),
            "expected_source_type": item.get("expected_source_type"),
            "expected_kind": item.get("expected_kind"),
            "positive_entity_ids": item.get("positive_entity_ids"),
            "positive_passage_ids": item.get("positive_passage_ids"),
        } for item in contextual_dataset["questions"]],
        "adaptive_dataset_id": adaptive_dataset["dataset_id"],
        "adaptive_dataset_provenance": adaptive_dataset.get("provenance"),
        "adaptive_questions": [{"id": item["id"], "question": item["question"],
                                "split": item["split"], "category": item.get("category")}
                               for item in adaptive_dataset["questions"]],
        "adaptive_challenge_dataset_id": challenge_dataset["dataset_id"],
        "adaptive_challenge_provenance": challenge_dataset.get("provenance"),
        "adaptive_challenge_questions": [{"id": item["id"], "question": item["question"],
                                          "split": item["split"], "category": item.get("category")}
                                         for item in challenge_dataset["questions"]],
        "relationship_dataset_id": relationship_dataset["dataset_id"],
        "relationship_challenge_dataset_id": relationship_dataset["challenge_dataset_id"],
        "relationship_questions": [{
            "id": item["id"], "question": item["question"], "split": item["split"],
            "type": item.get("type"), "gold_edge_ids": [edge["id"] for edge in item["gold_edges"]],
        } for item in relationship_dataset["cases"]],
        "graph_methods": [{"id": method, "label": {
            "text_hybrid": "Text hybrid baseline", "gcn_1": "Trainable GCN · 1 layer",
            "gcn_2": "Trainable GCN · 2 layers", "gcn_3": "Trainable GCN · 3 layers",
            "transe": "TransE", "transh": "TransH", "boxe": "BoxE",
        }[method]} for method in GRAPH_METHODS],
        "link_prediction_methods": [{"id": method, "label": LINK_PREDICTION_METHOD_LABELS[method]}
                                    for method in LINK_PREDICTION_METHODS],
        "link_prediction_graph": link_graph,
        "link_prediction_best": link_prediction_best,
        "log_path": str(RETRIEVAL_BENCHMARK_LOG_PATH),
    }


@app.get("/api/v1/retrieval-benchmark/runs")
def retrieval_benchmark_runs() -> dict:
    try:
        runs = list_saved_retrieval_runs()
    except ModelError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"runs": runs, "log_path": str(RETRIEVAL_BENCHMARK_LOG_PATH)}


@app.get("/api/v1/retrieval-benchmark/runs/{run_id}")
def saved_retrieval_benchmark_run(run_id: UUID) -> dict:
    try:
        result = load_saved_retrieval_run(str(run_id))
    except ModelError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="The saved retrieval benchmark run was not found.")
    return result


@app.post("/api/v1/retrieval-benchmark/stream")
async def retrieval_benchmark_stream(body: RetrievalBenchmarkRequest) -> StreamingResponse:
    if body.track != "adaptive_retrieval" and body.adaptive_question_set != "v2":
        raise HTTPException(status_code=422, detail="The adaptive question set applies only to adaptive retrieval.")
    if body.track != "link_prediction" and not body.models:
        raise HTTPException(status_code=422, detail="Select at least one embedding model.")
    if body.track in {"query_processing", "reranking", "adaptive_retrieval"} and len(body.models) != 1:
        raise HTTPException(status_code=422,
                            detail="Select exactly one embedding model for query, adaptive retrieval, or reranking experiments.")
    if body.track == "adaptive_retrieval" and (body.query_models or body.query_methods or body.query_method or body.rerankers):
        raise HTTPException(status_code=422, detail="Adaptive retrieval uses one query_model and a fixed first search; query methods and rerankers do not apply.")
    if body.track == "relationships" and not body.graph_methods:
        raise HTTPException(status_code=422, detail="Select at least one relationship retrieval method.")
    if body.track == "link_prediction" and (not body.graph_methods
            or "text_hybrid" in body.graph_methods):
        raise HTTPException(status_code=422, detail="Select at least one graph embedding method for link prediction.")
    if body.track == "link_prediction" and any(method.startswith("gcn_") for method in body.graph_methods) \
            and len(body.models) != 1:
        raise HTTPException(status_code=422,
                            detail="Select exactly one embedding model to provide GCN node features.")
    run_id = new_query_id()

    async def events():
        queue: asyncio.Queue[dict] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(item: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        async def run() -> None:
            try:
                if body.track == "relationships":
                    await asyncio.to_thread(
                        run_relationship_retrieval_benchmark, CATALOG, DATA_DIR, app.state.models,
                        body.models, body.question_ids, body.split, body.graph_methods,
                        app.state.relationship_benchmark_dataset, run_id,
                        app.state.model_run_lock, emit, append_retrieval_benchmark_run,
                    )
                elif body.track == "link_prediction":
                    await asyncio.to_thread(
                        run_link_prediction_benchmark, CATALOG, DATA_DIR, app.state.models,
                        body.models[0] if body.models else "", body.graph_methods, body.split, run_id,
                        app.state.model_run_lock, emit, append_retrieval_benchmark_run,
                    )
                elif body.track == "adaptive_retrieval":
                    await asyncio.to_thread(
                        run_adaptive_retrieval_benchmark, CATALOG, DATA_DIR, app.state.models,
                        body.models[0], body.question_ids, body.split,
                        (load_adaptive_retrieval_dataset(CHALLENGE_DATASET_PATH)
                         if body.adaptive_question_set == "reformulation_challenge_candidate"
                         else load_adaptive_retrieval_dataset()), run_id, app.state.model_run_lock,
                        emit, append_retrieval_benchmark_run,
                        query_model=body.query_model or app.state.models.router_model,
                    )
                elif body.track in {"query_processing", "reranking"}:
                    query_dataset = (app.state.catalog_object_retrieval_dataset
                                     if body.question_set == "catalog_objects"
                                     else app.state.contextual_retrieval_dataset
                                     if body.question_set == "contextual"
                                     else app.state.retrieval_benchmark_dataset)
                    await asyncio.to_thread(
                        run_query_retrieval_benchmark, CATALOG, DATA_DIR, app.state.models,
                        body.models[0], body.question_ids, body.split,
                        query_dataset, run_id,
                        app.state.model_run_lock, emit, append_retrieval_benchmark_run,
                        experiment=body.track, query_model=body.query_model or app.state.models.router_model,
                        query_models=body.query_models,
                        query_methods=body.query_methods, query_method=body.query_method,
                        rerankers=body.rerankers, question_set=body.question_set,
                    )
                else:
                    dataset = (app.state.catalog_object_retrieval_dataset
                               if body.track == "catalog_objects"
                               else app.state.retrieval_benchmark_dataset)
                    await asyncio.to_thread(
                        run_retrieval_benchmark, CATALOG, DATA_DIR, app.state.models,
                        body.models, body.question_ids, body.split, dataset,
                        run_id, app.state.model_run_lock, emit,
                    )
            except Exception as exc:
                logger.exception("Retrieval benchmark failed run_id=%s", run_id)
                await queue.put({"type": "retrieval_benchmark_error", "run_id": run_id,
                                 "detail": str(exc) if isinstance(exc, (ValueError, ModelError))
                                 else "The retrieval benchmark could not be completed."})

        worker = asyncio.create_task(run())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(item, separators=(',', ':'))}\n\n"
                if item["type"] in {"retrieval_benchmark_completed", "relationship_benchmark_completed",
                                    "link_prediction_benchmark_completed",
                                    "retrieval_benchmark_error"}:
                    break
            await worker
        except asyncio.CancelledError:
            raise

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Benchmark-ID": run_id},
    )
