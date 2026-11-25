#!/usr/bin/env python3
"""Export the saved cosine gate's test outputs after verifying catalog and labels.

This reads only local source-derived artifacts and never calls a model. It
validates the pinned index fingerprint, the development-only threshold, all
test labels and decisions, and the exact nearest fragment text before writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.corpus import load_catalog, render_chunks
from app.decision_router import initial_query
from app.graph import SchemaGraph
from app.index import EMBEDDING_FRAGMENT_CHARS, _split_embedding_text
from scripts.benchmark_cosine_router import choose_threshold, expected_search
from scripts.benchmark_router_decisions import accepted_routes, summarize, validate_dataset


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def catalog_fragments(result: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Reconstruct the exact indexed text only after verifying the index identity."""
    catalog = load_catalog(ROOT / "seed/catalog.json")
    SchemaGraph(catalog["entities"], catalog["source_commit"], semantics=catalog.get("semantics", {}))
    chunks = render_chunks(catalog)
    embedding = result["embedding"]
    payload = {"chunks": chunks, "model": embedding["name"], "model_digest": embedding["digest"],
               "format": "cdm-passages-v3", "embedding_fragment_chars": EMBEDDING_FRAGMENT_CHARS,
               "embedding_format": "nomic-search-v1"}
    identity = sha(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())
    if identity != result["index_identity"] or catalog["source_commit"] != result["source_commit"]:
        raise ValueError("The catalog and embedding do not reproduce the recorded index identity.")
    fragments: dict[str, tuple[str, str]] = {}
    for chunk in chunks:
        text = chunk["text"]
        if len(text) <= EMBEDDING_FRAGMENT_CHARS:
            parts = [text]
        else:
            prefix = f"{chunk.get('title') or chunk['id']}: "[:EMBEDDING_FRAGMENT_CHARS // 4]
            parts = [prefix + part for part in _split_embedding_text(text, EMBEDDING_FRAGMENT_CHARS - len(prefix))]
        for number, part in enumerate(parts, 1):
            fragment_id = chunk["id"] if len(parts) == 1 else f"{chunk['id']}::fragment:{number}"
            if fragment_id in fragments:
                raise ValueError("The catalog contains duplicate embedding fragment IDs.")
            fragments[fragment_id] = (chunk["id"], part)
    return fragments


def build_snapshot(result: dict[str, Any], source_sha: str) -> dict[str, Any]:
    if result.get("status") != "completed":
        raise ValueError("Only a complete cosine benchmark can be exported.")
    raw = {"routing": (ROOT / "evaluation/routing-decision-v1.json").read_bytes(),
           "scope": (ROOT / "evaluation/routing-scope-binary-v1.json").read_bytes(),
           "scope_source": (ROOT / "evaluation/scope-questions-v1.json").read_bytes()}
    if result.get("dataset_sha256") != {key: sha(value) for key, value in raw.items()}:
        raise ValueError("Recorded dataset hashes differ from the current source datasets.")
    routing, scope = (json.loads(raw[key]) for key in ("routing", "scope"))
    datasets = {item["dataset_id"]: item for item in (routing, scope)}
    fragments = catalog_fragments(result)
    selected = choose_threshold(result["development"]["rows"])
    if selected != result.get("calibration"):
        raise ValueError("The saved threshold does not reproduce from the development probes.")
    threshold = selected["threshold"]
    exported: dict[str, Any] = {}
    for dataset_id, dataset in datasets.items():
        cases = {case["id"]: case for case in validate_dataset(dataset) if case["split"] == "test"}
        group = result["datasets"][dataset_id]
        rows = group["rows"]
        if group.get("split") != "test" or len(rows) != len(cases) or {r["case_id"] for r in rows} != set(cases):
            raise ValueError("Saved cosine cases do not exactly cover the selected test split.")
        clean_rows = []
        for row in rows:
            case = cases[row["case_id"]]
            if (row["split"] != "test" or row["question"] != case["question"]
                    or row.get("context_messages", []) != case.get("context_messages", [])
                    or row["accepted_routes"] != accepted_routes(case)
                    or row["expected_search"] != expected_search(case) or row["query"] != initial_query(case)):
                raise ValueError("A saved cosine row does not match its dataset question or label.")
            if row["status"] == "scored":
                fragment = fragments.get(row["nearest_fragment_id"])
                if fragment != (row["nearest_passage_id"], row["nearest_fragment_text"]):
                    raise ValueError("Saved nearest-fragment metadata does not match the verified catalog.")
                predicted = "raw_search" if row["similarity"] >= threshold else "no_search"
                if row["predicted_route"] != predicted or row["correct"] != (predicted in row["accepted_routes"]):
                    raise ValueError("A saved decision does not match the development-selected threshold.")
            elif row["status"] != "error" or row.get("predicted_route") is not None:
                raise ValueError("Invalid saved cosine probe status.")
            keys = ("case_id", "split", "question", "context_messages", "expected_route", "accepted_routes",
                    "expected_search", "query", "status", "similarity", "nearest_fragment_id", "nearest_passage_id",
                    "nearest_fragment_text", "nearest_passage", "predicted_route", "correct", "error", "elapsed_seconds")
            clean_rows.append({key: row[key] for key in keys if key in row})
        if summarize(rows) != group["summary"]:
            raise ValueError("A saved test summary does not reproduce from its rows.")
        exported[dataset_id] = {"split": "test", "rows": clean_rows, "summary": group["summary"]}
    return {"schema_version": 1, "description": "Saved no-LLM nearest-fragment cosine gate on the test splits.",
            "method": result["method"], "caveat": result["caveat"],
            "artifact_sha256": source_sha, "dataset_sha256": result["dataset_sha256"],
            "source_commit": result["source_commit"], "index_identity": result["index_identity"],
            "embedding": {key: result["embedding"][key] for key in ("name", "digest")},
            "calibration": result["calibration"], "datasets": exported}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/macstudio-runs/routing-cosine-threshold-test.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results/routing-cosine-snapshot.json")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    snapshot = build_snapshot(json.loads(raw), sha(raw))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(snapshot, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
