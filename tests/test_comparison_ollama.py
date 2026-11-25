import argparse

import pytest

from app.ollama import Ollama
from scripts.comparison_ollama import ComparisonOllama, add_comparison_options, comparison_settings


def clients():
    args = ("http://unused", "embedding", "router", "answer")
    return Ollama(*args, judge_model="judge"), ComparisonOllama(*args, judge_model="judge")


@pytest.mark.parametrize("model,as_json,tokens", [("judge", True, 600), ("answer", False, 700)])
def test_default_matches_app_payload(model, as_json, tokens):
    baseline, comparison = clients()
    messages = [{"role": "user", "content": "fixed evidence"}]
    kwargs = {"stream": True, "as_json": as_json, "max_tokens": tokens, "schema": {"type": "object"}}
    assert comparison._chat_payload(model, messages, **kwargs) == baseline._chat_payload(model, messages, **kwargs)


@pytest.mark.parametrize("model,as_json,tokens", [("judge", True, 600), ("answer", False, 700)])
def test_reasoning_override_preserves_prompt_and_schema(model, as_json, tokens):
    baseline, _ = clients()
    comparison = ComparisonOllama("http://unused", "embedding", "router", "answer", judge_model="judge",
                                  reasoning_effort="low", max_output_tokens=2048)
    messages = [{"role": "user", "content": "fixed evidence" * 1000}]
    kwargs = {"stream": True, "as_json": as_json, "max_tokens": tokens, "schema": {"type": "object"}}
    old = baseline._chat_payload(model, messages, **kwargs)
    new = comparison._chat_payload(model, messages, **kwargs)
    assert new["messages"] == old["messages"] == messages
    assert new.get("format") == old.get("format")
    assert new["model"] == old["model"] and new["stream"] == old["stream"]
    assert new["think"] == "low" and new["options"]["num_predict"] == 2048
    assert new["options"]["temperature"] == old["options"]["temperature"] == 0
    assert new["options"]["num_ctx"] >= old["options"]["num_ctx"]


def test_cli_options_and_recorded_settings():
    parser = argparse.ArgumentParser()
    add_comparison_options(parser)
    defaults = parser.parse_args([])
    assert comparison_settings(defaults.reasoning_effort, defaults.max_output_tokens, 600)["think"] is False
    args = parser.parse_args(["--reasoning-effort", "low", "--max-output-tokens", "2048"])
    settings = comparison_settings(args.reasoning_effort, args.max_output_tokens, 700)
    assert settings["think"] == "low" and settings["max_output_tokens"] == 2048
    args = parser.parse_args(["--reasoning-effort", "on", "--max-output-tokens", "2048"])
    assert comparison_settings(args.reasoning_effort, args.max_output_tokens, 700)["think"] is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-output-tokens", "0"])


@pytest.mark.parametrize("model,as_json,tokens", [("judge", True, 600), ("answer", False, 700)])
def test_boolean_thinking_preserves_prompt_and_schema(model, as_json, tokens):
    baseline, _ = clients()
    comparison = ComparisonOllama("http://unused", "embedding", "router", "answer", judge_model="judge",
                                  reasoning_effort="on", max_output_tokens=2048)
    messages = [{"role": "user", "content": "fixed evidence" * 1000}]
    kwargs = {"stream": True, "as_json": as_json, "max_tokens": tokens, "schema": {"type": "object"}}
    old = baseline._chat_payload(model, messages, **kwargs)
    new = comparison._chat_payload(model, messages, **kwargs)
    assert new["messages"] == old["messages"] == messages
    assert new.get("format") == old.get("format")
    assert new["model"] == old["model"] and new["stream"] == old["stream"]
    assert new["think"] is True and new["options"]["num_predict"] == 2048
    assert new["options"]["temperature"] == old["options"]["temperature"] == 0
    assert new["options"]["num_ctx"] >= old["options"]["num_ctx"]


def test_explicit_context_cap_increases_judge_context_and_is_recorded():
    parser = argparse.ArgumentParser()
    add_comparison_options(parser)
    args = parser.parse_args(["--context-cap", "16384", "--max-output-tokens", "8192"])
    assert comparison_settings(None, args.max_output_tokens, 600, args.context_cap)["context_cap_override"] == 16384
    client = ComparisonOllama("http://unused", "embedding", "router", "answer", judge_model="judge",
                              max_output_tokens=args.max_output_tokens, context_cap=args.context_cap)
    payload = client._chat_payload("judge", [{"role": "user", "content": "x" * 12000}], stream=True,
                                   as_json=True, max_tokens=600, schema={"type": "object"})
    assert 8192 < payload["options"]["num_ctx"] <= 16384
    assert payload["options"]["num_predict"] == 8192


def test_context_cap_validation():
    parser = argparse.ArgumentParser()
    add_comparison_options(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--context-cap", "1024"])
    args = ("http://unused", "embedding", "router", "answer")
    for cap in (True, 4095, 8192.0):
        with pytest.raises(ValueError):
            ComparisonOllama(*args, context_cap=cap)
    with pytest.raises(ValueError):
        ComparisonOllama(*args, context_cap=8192, max_output_tokens=8192)
