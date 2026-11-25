"""Optional hosted, typed query router for the CDM retrieval pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import httpx

from .ollama import ModelError


ROUTE_CHOICES = frozenset({"raw_search", "rewrite_search", "no_search"})


@dataclass(frozen=True)
class RouteDecision:
    choice: str
    confidence: float
    probabilities: dict[str, float]
    model: str
    input_tokens: int
    output_tokens: int


class JevRouter:
    """Classify whether to search as written, rewrite locally, or skip search."""

    def __init__(self, api_key: str, model: str = "jev-latest",
                 base_url: str = "https://api.typesafe.ai", min_confidence: float = 0.70,
                 timeout: float = 10.0):
        if not api_key.strip():
            raise ValueError("CDM_RAG_ROUTER_BACKEND=jev requires JEV_API_KEY.")
        if not model.strip() or not base_url.strip():
            raise ValueError("The Jev router model and API URL must be non-empty.")
        if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("CDM_RAG_ROUTER_MIN_CONFIDENCE must be between 0 and 1.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.min_confidence = min_confidence
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def classify(self, question: str) -> RouteDecision:
        payload = {
            "model": self.model,
            "state": question,
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": (
                        "Choose the best retrieval route for this one message. The collection contains extracted "
                        "Microsoft CDM knowledge, including entity definitions, attributes, relationships, "
                        "manifests, traits, inheritance, and metadata, not real customer account records."
                    ),
                    "criteria": {
                        "raw_search": (
                            "The message is an in-scope CDM question about entities, attributes, relationships, "
                            "manifest contents, traits, inheritance, or banking schema whose original wording "
                            "can be sent directly to hybrid text retrieval."
                        ),
                        "rewrite_search": (
                            "The message is an in-scope CDM question about entities, attributes, relationships, "
                            "manifest contents, traits, inheritance, or banking schema, but its indirect or "
                            "conceptual wording would benefit from a short natural-language search rewrite."
                        ),
                        "no_search": (
                            "The message is unrelated, a greeting or personal story, requests actual personal "
                            "account data, or asks for information absent from this extracted CDM collection."
                        ),
                    },
                }
            },
        }
        try:
            response = self.client.post(
                self.base_url + "/v1/systemone",
                headers={"Authorization": "Bearer " + self.api_key},
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelError("The Jev routing service could not complete the request.") from exc
        if not isinstance(result, dict):
            raise ModelError("The Jev routing service returned an invalid response.")
        answers = result.get("answers")
        answer = answers.get("route") if isinstance(answers, dict) else None
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ModelError("The Jev routing service returned an invalid route decision.")
        choice = answer.get("choice")
        confidence = answer.get("confidence")
        probabilities = answer.get("probabilities")
        if (not isinstance(choice, str) or choice not in ROUTE_CHOICES
                or not isinstance(confidence, (int, float))
                or isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 <= confidence <= 1
                or not isinstance(probabilities, dict)):
            raise ModelError("The Jev routing service returned an invalid route decision.")
        clean_probabilities: dict[str, float] = {}
        for label, probability in probabilities.items():
            if (label not in ROUTE_CHOICES or not isinstance(probability, (int, float))
                    or isinstance(probability, bool) or not math.isfinite(probability)
                    or not 0 <= probability <= 1):
                raise ModelError("The Jev routing service returned invalid route probabilities.")
            clean_probabilities[label] = float(probability)
        usage = result.get("usage")
        if not isinstance(usage, dict):
            raise ModelError("The Jev routing service returned invalid usage details.")
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        if (not isinstance(input_tokens, int) or isinstance(input_tokens, bool) or input_tokens < 0
                or not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens < 0):
            raise ModelError("The Jev routing service returned invalid usage details.")
        model = result.get("model", self.model)
        if not isinstance(model, str) or not model:
            raise ModelError("The Jev routing service returned an invalid model name.")
        return RouteDecision(choice, float(confidence), clean_probabilities, model,
                             input_tokens, output_tokens)
