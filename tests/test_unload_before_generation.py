import httpx
import pytest

from app.runtime import ollama
from app.runtime.ollama import ModelError, Ollama


def model_client():
    return Ollama("http://local", "embed", "rewrite", "gemma4:12b")


def test_shared_pool_never_controls_residency(monkeypatch):
    monkeypatch.setenv("CDM_RAG_MANAGE_MODEL_RESIDENCY", "false")
    client = Ollama("http://pool", "qwen3-embedding:0.6b", "rewrite", "answer")
    payloads = []
    monkeypatch.setattr(client, "_post", lambda path, payload, **kwargs: payloads.append(payload) or {"embeddings": [[1.0]]})
    client.unload_if_loaded("rewrite")
    client.unload_other_models("answer")
    assert not payloads
    client.embed(["question"], query=True, keep_alive=0)
    assert "keep_alive" not in payloads[0]


def fake_resident_client(monkeypatch, models):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": list(models)}

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            return Response()

    monkeypatch.setattr(httpx, "Client", Client)


def test_unload_others_preserves_answer_aliases_and_unloads_embedding(monkeypatch):
    models = [{"name": "gemma4:12b", "digest": "answer"},
              {"name": "alias:latest", "model": "alias:latest", "digest": "answer"},
              {"name": "rewrite:latest", "digest": "router"},
              {"model": "embed:latest", "digest": "embedding"}]
    fake_resident_client(monkeypatch, models)
    client = model_client()
    unloaded = []

    def unload(path, payload, timeout):
        assert path == "/api/generate" and payload["keep_alive"] == 0
        name = payload["model"]
        unloaded.append(name)
        models[:] = [item for item in models if (item.get("name") or item.get("model")) != name]
        return {}

    monkeypatch.setattr(client, "_post", unload)
    client.unload_other_models("gemma4:12b")
    assert unloaded == ["rewrite:latest", "embed:latest"]
    assert {item.get("digest") for item in models} == {"answer"}


def test_unload_does_not_stop_latest_alias_of_answer(monkeypatch):
    models = [{"name": "answer:latest"}]
    fake_resident_client(monkeypatch, models)
    client = model_client()
    monkeypatch.setattr(client, "_post", lambda *args, **kwargs: pytest.fail("answer was unloaded"))
    client.unload_other_models("answer")


def test_unload_partial_failure_is_surfaced(monkeypatch):
    models = [{"name": "one"}, {"name": "two"}, {"name": "gemma4:12b"}]
    fake_resident_client(monkeypatch, models)
    client = model_client()

    def unload(path, payload, timeout):
        if payload["model"] == "two":
            raise ModelError("server failed")
        models[:] = [item for item in models if item["name"] != payload["model"]]
        return {}

    monkeypatch.setattr(client, "_post", unload)
    with pytest.raises(ModelError, match="two.*before answer generation"):
        client.unload_other_models("gemma4:12b")
    assert [item["name"] for item in models] == ["two", "gemma4:12b"]


def test_unload_rejects_unavailable_process_list(monkeypatch):
    class BadClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            raise httpx.ConnectError("unavailable")

    monkeypatch.setattr(httpx, "Client", BadClient)
    with pytest.raises(ModelError, match="list loaded models before answer generation"):
        model_client().unload_other_models("gemma4:12b")


def test_unload_requires_ollama_to_report_the_model_gone(monkeypatch):
    models = [{"name": "router"}]
    fake_resident_client(monkeypatch, models)
    client = model_client()
    monkeypatch.setattr(client, "_post", lambda *args, **kwargs: {})
    times = iter((0, 11))
    monkeypatch.setattr(ollama.time, "monotonic", lambda: next(times))
    with pytest.raises(ModelError, match="Models still loaded.*router"):
        client.unload_other_models("gemma4:12b")
