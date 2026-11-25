"""Preserve user wording when a model reformulates a search query."""
from __future__ import annotations

import re
from typing import Any


# Only carry a previous user turn when the latest question refers back to it.
# Assistant messages may contain unverified claims and are deliberately excluded.
_REFERENCE = re.compile(
    r"\b(?:it|its|they|them|their|theirs|this|that|these|those|former|latter)\b", re.I
)


def anchor_search_query(query: str, question: str,
                        context_messages: list[dict[str, Any]] | None = None) -> str:
    """Keep the original question and bounded, user-supplied antecedent in the query.

    The anchor is only search input; it is not a source of facts for the answer.
    It supplements the model rewrite and does not assert that the rewrite is true.
    """
    rewrite = " ".join(query.split())
    current = " ".join(question.split())
    components = [rewrite]
    if current and current.casefold() not in rewrite.casefold():
        components.append(current)
    if _REFERENCE.search(current):
        for message in reversed(context_messages or []):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            # Latest previous user turn is bounded; no earlier unrelated turns or
            # assistant-generated statements enter the retrieval query.
            antecedent = " ".join(content.split())[:1200]
            if antecedent and antecedent.casefold() not in " ".join(components).casefold():
                components.append(antecedent)
            break
    return "\n".join(components)
