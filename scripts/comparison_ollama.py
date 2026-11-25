"""Opt-in model settings for isolated saved-evidence comparisons only."""
from __future__ import annotations

import argparse
from typing import Any

from app.ollama import Ollama


def positive_tokens(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def context_cap_tokens(value: str) -> int:
    number = positive_tokens(value)
    if number < 4096:
        raise argparse.ArgumentTypeError("context cap must be at least 4096 tokens")
    return number


def add_comparison_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reasoning-effort", choices=("on", "low", "medium", "high"),
                        help="Use 'on' for boolean thinking or low/medium/high for models supporting effort levels; "
                             "omitted preserves the app default")
    parser.add_argument("--max-output-tokens", type=positive_tokens,
                        help="Override output token budget; omitted preserves the app default")
    parser.add_argument("--context-cap", type=context_cap_tokens,
                        help="Experimental maximum context tokens; omitted preserves role-based app limits")


def comparison_settings(reasoning_effort: str | None, max_output_tokens: int | None,
                        default_max_output_tokens: int, context_cap: int | None = None) -> dict[str, Any]:
    return {"think": True if reasoning_effort == "on" else reasoning_effort if reasoning_effort is not None else False,
            "max_output_tokens": max_output_tokens if max_output_tokens is not None else default_max_output_tokens,
            "max_output_tokens_override": max_output_tokens,
            "context_cap_override": context_cap,
            "temperature": 0,
            "context_policy": ("explicit cap and input-size estimate, using effective output token budget"
                               if context_cap is not None else
                               "app role-based cap and input-size estimate, using effective output token budget")}


class ComparisonOllama(Ollama):
    """Change only generation settings, keeping prompts and source material intact."""

    def __init__(self, *args: Any, reasoning_effort: str | None = None,
                 max_output_tokens: int | None = None, context_cap: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if reasoning_effort not in (None, "on", "low", "medium", "high"):
            raise ValueError("Invalid reasoning effort")
        if max_output_tokens is not None and (isinstance(max_output_tokens, bool) or max_output_tokens < 1):
            raise ValueError("Output token budget must be a positive integer")
        if context_cap is not None and (isinstance(context_cap, bool) or not isinstance(context_cap, int) or context_cap < 4096):
            raise ValueError("Context cap must be an integer of at least 4096 tokens")
        if context_cap is not None and max_output_tokens is not None and max_output_tokens >= context_cap:
            raise ValueError("Output token budget must be smaller than context cap")
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.chat_context_cap_override = context_cap

    def _chat_payload(self, model: str, messages: list[dict[str, str]], *, stream: bool,
                      as_json: bool, max_tokens: int, schema: dict[str, Any] | None) -> dict[str, Any]:
        effective_tokens = self.max_output_tokens if self.max_output_tokens is not None else max_tokens
        payload = super()._chat_payload(model, messages, stream=stream, as_json=as_json,
                                        max_tokens=effective_tokens, schema=schema)
        if self.reasoning_effort is not None:
            payload["think"] = True if self.reasoning_effort == "on" else self.reasoning_effort
        return payload
