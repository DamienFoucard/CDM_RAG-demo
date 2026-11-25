#!/usr/bin/env python3
"""Create a blinded manual-review worksheet from fixed-evidence answer runs.

The input and output can contain private questions and answer text. The worksheet
omits model and retrieval-arm names; its separate key must not go to the reviewer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import secrets
from pathlib import Path
from typing import Any


def _fenced(text: Any) -> str:
    value = str(text if text is not None else "")
    longest = max((len(run) for run in re.findall(r"`+", value)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{value}\n{fence}"


def make_review(payload: dict[str, Any], seed: int) -> tuple[str, dict[str, Any]]:
    cases: dict[tuple[str, str], dict[str, Any]] = {}
    for case in payload.get("cases", []):
        key = (case["case_id"], case["arm"])
        if key in cases:
            raise ValueError(f"Duplicate case/arm: {key}")
        cases[key] = case
    rows = payload.get("rows")
    if not cases or not isinstance(rows, list) or not rows:
        raise ValueError("Input must contain cases and answer rows")
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    output = [
        "# Blind answer evidence review", "",
        "For each answer, check every factual claim against its cited passage, using the complete passage text below. "
        "Evaluate requested coverage separately from factual support. The reference is an incomplete, provisional "
        "checklist, not authority; a fact absent from retrieved evidence may still be true in the source catalog.", "",
        "Mark each item as supported, unsupported, or uncertain; note omissions, wrong model or citation, "
        "and cases where overlapping target types are mistaken for the same record. "
        "An evidence-aware limitation is not itself an unsupported factual claim.", "",
    ]
    key_rows = []
    for position, row in enumerate(rows, 1):
        key = (row["case_id"], row["arm"])
        case = cases.get(key)
        if case is None:
            raise ValueError(f"Missing case/arm: {key}")
        evidence = {"contexts": case.get("contexts"), "sources": case.get("sources")}
        digest = hashlib.sha256(json.dumps(evidence, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
        if row.get("evidence_sha256") != digest or case.get("evidence_sha256") != digest:
            raise ValueError(f"Evidence hash mismatch: {key}")
        sources = case.get("sources")
        if not isinstance(sources, list):
            raise ValueError(f"Missing source passages: {key}")
        labels = [source.get("citation") for source in sources]
        if any(not isinstance(label, str) or not re.fullmatch(r"R\d+", label) for label in labels) or len(set(labels)) != len(labels):
            raise ValueError(f"Invalid or duplicate source labels: {key}")
        review_id = f"item-{position:03d}"
        answer = row.get("raw_answer") or ""
        output.extend([
            f"## {review_id}", "", "**Question**", "", _fenced(case.get("question", "")), "",
            "**Reference checklist**", "", _fenced(case.get("reference_answer", "")), "",
            "**Answer**", "", _fenced(answer if row.get("status") == "generated" else
                                      "[Answer unavailable; inspect the status in the key after review.]"), "",
            "**Retrieved passages**", "",
        ])
        for source in sources:
            metadata = {field: source[field] for field in ("source_id", "entity_id", "kind", "uri")
                        if source.get(field)}
            output.extend([f"- **[{source['citation']}]**", "", _fenced(json.dumps(metadata, ensure_ascii=False)),
                           "", _fenced(source.get("text", "")), ""])
        output.extend([
            "**Review**", "", "- Claim support: supported / unsupported / uncertain / answer unavailable",
            "- Requested coverage: complete / partial / not answered / uncertain",
            "- Citation accuracy and model-qualified names:",
            "- Specific unsupported claims and the evidence checked:",
            "- Missing requested facts and whether the answer acknowledged missing evidence:", "",
        ])
        key_rows.append({"review_id": review_id, "case_id": row["case_id"], "arm": row["arm"],
                         "model": row.get("model"), "status": row.get("status"),
                         "evidence_sha256": case["evidence_sha256"]})
    return "\n".join(output) + "\n", {"seed": seed, "items": key_rows}


def _write_new(path: Path, text: str) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        out.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--review", type=Path, required=True, help="New Markdown worksheet path")
    parser.add_argument("--key", type=Path, required=True, help="New private JSON mapping path")
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible shuffle seed")
    args = parser.parse_args()
    if args.review.resolve() == args.key.resolve():
        parser.error("Review and key must use different paths")
    raw = args.input.read_bytes()
    worksheet, key = make_review(json.loads(raw), args.seed if args.seed is not None else secrets.randbits(128))
    key["input_sha256"] = hashlib.sha256(raw).hexdigest()
    for path in (args.review, args.key):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_new(args.key, json.dumps(key, ensure_ascii=False, indent=2) + "\n")
    try:
        _write_new(args.review, worksheet)
    except Exception:
        args.key.unlink()
        raise
    print(f"Wrote {len(key['items'])} blinded items to {args.review}; keep {args.key} separate.")


if __name__ == "__main__":
    main()
