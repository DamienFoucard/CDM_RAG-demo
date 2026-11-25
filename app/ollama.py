"""Small local model client; failures are explicit and never fabricate evidence."""
from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from .query_trace import trace_embedding


class ModelError(RuntimeError):
    pass


def chat_context_size(messages: list[dict[str, str]], max_tokens: int, cap: int) -> int:
    """Apply the model client's prompt-size estimate and context cap."""
    prompt_chars = sum(len(message.get("content", "")) for message in messages)
    requested_context = (prompt_chars + 2) // 3 + max_tokens + 512
    return min(cap, max(4096, ((requested_context + 1023) // 1024) * 1024))


class Ollama:
    def __init__(self, base_url: str, embedding_model: str, router_model: str, answer_model: str,
                 scope_model: str = "qwen3:4b", judge_model: str = "qwen3:4b",
                 request_timeout: float | None = None, stream_idle_timeout: float | None = None,
                 first_token_timeout: float | None = None):
        self.base_url = base_url.rstrip("/")
        self.manage_model_residency = os.getenv("CDM_RAG_MANAGE_MODEL_RESIDENCY", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.embedding_model = embedding_model
        self.router_model = router_model
        # The standalone decision classifier may use a different local model
        # from the query rewriter. Legacy callers retain the same default.
        self.decision_model = router_model
        self.answer_model = answer_model
        self.scope_model = scope_model
        self.judge_model = judge_model
        # Benchmarks may set a uniform context policy on their private client.
        self.chat_context_cap_override: int | None = None
        self.query_embedding_keep_alive = os.getenv("OLLAMA_QUERY_EMBEDDING_KEEP_ALIVE", "0").strip() or "0"
        self.request_timeout = request_timeout or float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "300"))
        self.stream_idle_timeout = stream_idle_timeout or float(os.getenv("OLLAMA_STREAM_IDLE_TIMEOUT_SECONDS", "30"))
        self.first_token_timeout = first_token_timeout or float(os.getenv("OLLAMA_FIRST_TOKEN_TIMEOUT_SECONDS", "300"))
        self._chat_local = threading.local()

    def _post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=timeout or self.request_timeout, trust_env=False) as client:
                response = client.post(self.base_url + path, json=payload)
                response.raise_for_status()
                result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            if isinstance(exc, httpx.TimeoutException):
                raise ModelError("The local model service timed out before completing the request.") from exc
            raise ModelError("The local model service could not complete the request.") from exc
        if not isinstance(result, dict):
            raise ModelError("The local model service returned an invalid response.")
        return result

    def embedding_digest(self) -> str:
        try:
            with httpx.Client(timeout=10, trust_env=False) as client:
                response = client.get(self.base_url + "/api/tags")
                response.raise_for_status()
                models = response.json()["models"]
            requested = self.embedding_model.removesuffix(":latest")
            matches = [model for model in models
                       if str(model.get("name", "")).removesuffix(":latest") == requested]
            digest = matches[0].get("digest") if len(matches) == 1 else None
            if not isinstance(digest, str) or not digest:
                raise ValueError("embedding model is not installed")
            return digest
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise ModelError(f"Embedding model {self.embedding_model!r} is not available in Ollama.") from exc

    def unload_if_loaded(self, model: str) -> None:
        if not self.manage_model_residency:
            return
        """Unload a model already resident in Ollama without loading it just to stop it."""
        try:
            with httpx.Client(timeout=10, trust_env=False) as client:
                response = client.get(self.base_url + "/api/ps")
                response.raise_for_status()
                body = response.json()
            if not isinstance(body, dict):
                raise ValueError("invalid Ollama process list")
            loaded = body.get("models", [])
            if not isinstance(loaded, list):
                raise ValueError("invalid Ollama model list")
            requested = model.removesuffix(":latest")
            if not any(isinstance(item, dict)
                       and str(item.get("name", "")).removesuffix(":latest") == requested for item in loaded):
                return
            self._post("/api/generate", {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
                       timeout=10)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise ModelError(f"Could not unload model {model!r} from Ollama.") from exc

    def unload_other_models(self, keep_model: str) -> None:
        """Free resident Ollama models before answer generation, retaining the answer model.

        The caller must hold the application's model-run lock. Ollama does not
        expose a cross-client lease, so outside users of the server are not
        coordinated by that lock.
        """
        if not self.manage_model_residency:
            return

        def resident() -> list[dict[str, Any]]:
            try:
                with httpx.Client(timeout=10, trust_env=False) as client:
                    response = client.get(self.base_url + "/api/ps")
                    response.raise_for_status()
                    body = response.json()
                items = body.get("models") if isinstance(body, dict) else None
                if not isinstance(items, list) or any(
                    not isinstance(item, dict)
                    or not isinstance(item.get("name") or item.get("model"), str)
                    or not (item.get("name") or item.get("model"))
                    for item in items
                ):
                    raise ValueError("invalid Ollama model list")
                return items
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                raise ModelError("Could not list loaded models before answer generation.") from exc

        def names(item: dict[str, Any]) -> set[str]:
            return {value.removesuffix(":latest") for value in (item.get("name"), item.get("model"))
                    if isinstance(value, str) and value}

        loaded = resident()
        keep_name = keep_model.removesuffix(":latest")
        keep_digests = {item.get("digest") for item in loaded if keep_name in names(item)
                        and isinstance(item.get("digest"), str) and item.get("digest")}

        def should_unload(item: dict[str, Any]) -> bool:
            return keep_name not in names(item) and item.get("digest") not in keep_digests

        targets = [item for item in loaded if should_unload(item)]
        attempted: set[str] = set()
        for item in targets:
            model = item.get("name") or item["model"]
            if model.removesuffix(":latest") in attempted:
                continue
            attempted.add(model.removesuffix(":latest"))
            try:
                self._post("/api/generate", {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
                           timeout=10)
            except ModelError as exc:
                raise ModelError(f"Could not unload model {model!r} before answer generation.") from exc

        if not targets:
            return
        deadline = time.monotonic() + 10
        while True:
            remaining = [item for item in resident() if should_unload(item)]
            if not remaining:
                return
            if time.monotonic() >= deadline:
                models = ", ".join(str(item.get("name") or item.get("model")) for item in remaining)
                raise ModelError(f"Models still loaded before answer generation: {models}.")
            time.sleep(0.1)

    def runtime_info(self, model: str) -> dict[str, Any] | None:
        """Report the active Ollama runner and memory for one configured model."""
        try:
            with httpx.Client(timeout=2, trust_env=False) as client:
                response = client.get(self.base_url + "/api/ps")
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        loaded = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(loaded, list):
            return None
        requested = model.removesuffix(":latest")
        active = next((item for item in loaded if isinstance(item, dict)
                       and str(item.get("name", item.get("model", ""))).removesuffix(":latest")
                       == requested), None)
        if active is None:
            return None
        return {
            "runner": active.get("runner") if isinstance(active.get("runner"), str) else None,
            "size_bytes": active.get("size") if isinstance(active.get("size"), int) else None,
            "gpu_bytes": active.get("size_vram") if isinstance(active.get("size_vram"), int) else None,
            "context_length": (active.get("context_length")
                               if isinstance(active.get("context_length"), int) else None),
        }

    def embedding_format(self) -> str:
        family = self.embedding_model.split(":", 1)[0]
        if family == "nomic-embed-text":
            return "nomic-search-v1"
        if family == "qwen3-embedding":
            return "qwen3-cdm-retrieval-v2"
        raise ModelError("The embedding model has no configured query/document format.")

    @trace_embedding
    def embed(self, texts: list[str], *, query: bool,
              keep_alive: str | int | None = None) -> list[list[float]]:
        encoding = self.embedding_format()
        if encoding == "nomic-search-v1":
            prefix = "search_query: " if query else "search_document: "
        else:
            prefix = ("Instruct: Retrieve passages from the extracted Microsoft CDM catalog that answer the user's question.\nQuery: "
                      if query else "")
        payload: dict[str, Any] = {"model": self.embedding_model,
                                   "input": [prefix + text for text in texts], "truncate": False}
        if self.manage_model_residency and keep_alive is not None:
            payload["keep_alive"] = keep_alive
        elif self.manage_model_residency and query:
            payload["keep_alive"] = self.query_embedding_keep_alive
        result = self._post("/api/embed", payload)
        vectors = result.get("embeddings")
        if (not isinstance(vectors, list) or len(vectors) != len(texts)
                or any(not isinstance(vector, list) or not vector
                       or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                              for value in vector) for vector in vectors)):
            raise ModelError("The embedding model returned incomplete or invalid vectors.")
        if len({len(vector) for vector in vectors}) != 1:
            raise ModelError("The embedding model returned inconsistent vector dimensions.")
        return vectors

    def _chat_payload(self, model: str, messages: list[dict[str, str]], *, stream: bool,
                      as_json: bool, max_tokens: int, schema: dict[str, Any] | None) -> dict[str, Any]:
        context_cap = (self.chat_context_cap_override if self.chat_context_cap_override is not None else
                       8192 if model == self.judge_model else (32768 if model == self.answer_model else 4096))
        context_size = chat_context_size(messages, max_tokens, context_cap)
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream,
                                   "think": False, "options": {"temperature": 0, "num_predict": max_tokens,
                                                                 "num_ctx": context_size}}
        if as_json:
            payload["format"] = schema or "json"
        return payload

    def chat(self, model: str, messages: list[dict[str, str]], *, as_json: bool = False,
             max_tokens: int = 600, schema: dict[str, Any] | None = None) -> str:
        return self._stream_chat(model, messages, on_token=lambda _: None, max_tokens=max_tokens,
                                 as_json=as_json, schema=schema)

    def chat_stream(self, model: str, messages: list[dict[str, str]], *, on_token: Callable[[str], None],
                    max_tokens: int = 700) -> str:
        """Stream answer text, failing only after an idle gap without model output."""
        return self._stream_chat(model, messages, on_token=on_token, max_tokens=max_tokens,
                                 as_json=False, schema=None)

    def _stream_chat(self, model: str, messages: list[dict[str, str]], *, on_token: Callable[[str], None],
                     max_tokens: int, as_json: bool, schema: dict[str, Any] | None) -> str:
        self._chat_local.timings = None
        payload = self._chat_payload(model, messages, stream=True, as_json=as_json,
                                     max_tokens=max_tokens, schema=schema)
        content: list[str] = []
        native_timings: dict[str, int] = {}
        content_chars = 0
        request_started_at = time.monotonic()
        last_token_at: float | None = None
        line_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        reader_thread: threading.Thread | None = None
        try:
            timeout = httpx.Timeout(
                self.first_token_timeout,
                connect=min(self.first_token_timeout, 10.0),
            )
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                with client.stream("POST", self.base_url + "/api/chat", json=payload) as response:
                    response.raise_for_status()
                    def read_lines() -> None:
                        try:
                            for stream_line in response.iter_lines():
                                line_queue.put(("line", (stream_line, time.monotonic())))
                        except BaseException as exc:
                            line_queue.put(("error", exc))
                        else:
                            line_queue.put(("done", None))

                    reader_thread = threading.Thread(target=read_lines, daemon=True)
                    reader_thread.start()
                    while True:
                        timeout_seconds = (self.stream_idle_timeout if last_token_at is not None
                                           else self.first_token_timeout)
                        timer_started_at = last_token_at if last_token_at is not None else request_started_at
                        remaining = timeout_seconds - (time.monotonic() - timer_started_at)
                        if remaining <= 0:
                            if last_token_at is None:
                                raise ModelError(f"No first token arrived within {self.first_token_timeout:g} seconds.")
                            raise ModelError(f"The local model produced no output for {self.stream_idle_timeout:g} seconds.")
                        try:
                            kind, item = line_queue.get(timeout=remaining)
                        except queue.Empty as exc:
                            if last_token_at is None:
                                raise ModelError(
                                    f"No first token arrived within {self.first_token_timeout:g} seconds."
                                ) from exc
                            raise ModelError(
                                f"The local model produced no output for {self.stream_idle_timeout:g} seconds."
                            ) from exc
                        if kind == "done":
                            break
                        if kind == "error":
                            if isinstance(item, httpx.TimeoutException):
                                if last_token_at is None:
                                    raise ModelError(
                                        f"No first token arrived within {self.first_token_timeout:g} seconds."
                                    ) from item
                                raise ModelError(
                                    f"The local model produced no output for {self.stream_idle_timeout:g} seconds."
                                ) from item
                            if isinstance(item, Exception):
                                raise item
                            raise ModelError("The local model stream reader failed.")
                        line, token_received_at = item
                        if not line:
                            continue
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ModelError("The local model returned an invalid streamed response.")
                        for key in ("total_duration", "load_duration", "prompt_eval_count",
                                    "prompt_eval_duration", "eval_count", "eval_duration"):
                            value = record.get(key)
                            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                                native_timings[key] = value
                        message = record.get("message", {})
                        # Reasoning tokens show that the model is still active,
                        # even though they are not part of the answer and must
                        # not be forwarded to the caller.
                        thinking = message.get("thinking", "") if isinstance(message, dict) else ""
                        if isinstance(thinking, str) and thinking:
                            last_token_at = token_received_at
                        token = message.get("content", "") if isinstance(message, dict) else ""
                        if token:
                            if not isinstance(token, str):
                                raise ModelError("The local model returned an invalid streamed token.")
                            content.append(token)
                            content_chars += len(token)
                            if content_chars > 20000:
                                raise ModelError("The local model returned an oversized answer.")
                            on_token(token)
                            last_token_at = token_received_at
                        if record.get("error"):
                            raise ModelError("The local model reported an error while generating the answer.")
        except httpx.TimeoutException as exc:
            raise ModelError("The local model request timed out before producing output.") from exc
        except httpx.ReadError as exc:
            raise ModelError("The local model stream was interrupted.") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelError("The local model service could not complete the streamed request.") from exc
        finally:
            if reader_thread is not None:
                reader_thread.join(timeout=0.2)
            self._chat_local.timings = native_timings or None
        answer = "".join(content).strip()
        if not answer:
            raise ModelError("The local model returned an empty answer.")
        return answer

    def chat_json(self, model: str, messages: list[dict[str, str]], *,
                  schema: dict[str, Any] | None = None, max_tokens: int = 300) -> dict[str, Any]:
        value, _ = self.chat_json_timed(model, messages, schema=schema, max_tokens=max_tokens)
        return value

    def chat_json_timed(self, model: str, messages: list[dict[str, str]], *,
                        schema: dict[str, Any] | None = None, max_tokens: int = 300
                        ) -> tuple[dict[str, Any], dict[str, int] | None]:
        """Return structured output plus Ollama's native duration/token counters."""
        try:
            value = json.loads(self.chat(model, messages, as_json=True, max_tokens=max_tokens, schema=schema))
        except json.JSONDecodeError as exc:
            raise ModelError("The local model returned invalid JSON.") from exc
        if not isinstance(value, dict):
            raise ModelError("The local model returned invalid JSON.")
        timings = getattr(self._chat_local, "timings", None)
        return value, timings
