#!/usr/bin/env python3
"""Export the saved test-run rankings and verified pinned-catalog text for the UI.

Contextual runs save up to six passage IDs or deduplicated entity IDs. Adaptive
runs also save the initial/follow-up top-20 lists, so their final ranking can be
reconstructed using the same reciprocal-rank fusion and checked against the
recorded top six. No searches or model calls are performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.adaptive_retrieval_benchmark import fuse_passages
from app.schema.corpus import load_catalog, render_chunks
from app.schema.graph import SchemaGraph
from app.retrieval.index import EMBEDDING_FRAGMENT_CHARS

RUNS = ROOT / "data/macstudio-runs"
OUTPUT = ROOT / "evaluation/results/retrieval-historical-output-snapshot.json"
RUN_FILES = {
    "context": "query-test-anchored-events.sse",
    "adaptive-4b": "adaptive-test-4b-events.sse",
    "adaptive-9b": "adaptive-test-9b-events.sse",
    "adaptive-12b": "adaptive-test-12b-events.sse",
}
EXPECTED_INDEX = "cd491dd029b4b4bc1e13e924ab8f1f66d1447f938fd03f70dcd4cd9a8d96775f"
MODEL_DIGEST = "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"


def load_run(path: Path) -> dict:
    events = [json.loads(line[6:]) for line in path.read_text().splitlines()
              if line.startswith("data: ")]
    found = [event["result"] for event in events
             if event.get("type") == "retrieval_benchmark_completed"]
    if len(found) != 1 or found[0].get("split") != "test":
        raise ValueError(f"Expected one completed test run: {path}")
    return found[0]


def main() -> None:
    challenge = json.loads((ROOT / "evaluation/reformulation-challenge-candidate-v1.json").read_text())
    catalog = load_catalog(ROOT / "seed/catalog.json")
    # Match SchemaIndex exactly: its graph constructor enriches the catalog.
    SchemaGraph(catalog["entities"], catalog["source_commit"],
                semantics=catalog.get("semantics", {}))
    chunks = render_chunks(catalog)
    identity = hashlib.sha256(json.dumps({
        "chunks": chunks, "model": "nomic-embed-text:latest",
        "model_digest": MODEL_DIGEST, "format": "cdm-passages-v3",
        "embedding_fragment_chars": EMBEDDING_FRAGMENT_CHARS,
        "embedding_format": "nomic-search-v1",
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if identity != EXPECTED_INDEX:
        raise ValueError("Current catalog and rendering do not match saved index identity")
    by_id = {chunk["id"]: chunk for chunk in chunks}
    runs = {name: load_run(RUNS / path) for name, path in RUN_FILES.items()}
    for run in runs.values():
        if (run["source_commit"] != catalog["source_commit"] or
                run["summaries"][0]["index_identity"] != identity):
            raise ValueError("Saved run and pinned catalog/index do not match")

    strategies: dict[str, list[dict]] = {}
    needed: set[str] = set()
    context_variants = {
        "context-raw": "raw",
        "context-planner-4b": "planner::qwen3:4b",
        "context-planner-9b": "planner::qwen3.5:9b-q4_K_M",
        "context-rewrite-4b": "rewrite::qwen3:4b",
        "context-rewrite-9b": "rewrite::qwen3.5:9b-q4_K_M",
    }
    for name, variant in context_variants.items():
        out = []
        for row in runs["context"]["rows"]:
            value = row["variants"][variant]
            passage_mode = "ranked_passage_ids" in value
            key = "ranked_passage_ids" if passage_mode else "ranked_entity_ids"
            matched = "matched_passage_ids" if passage_mode else "matched_entity_ids"
            ids = value[key]
            if passage_mode:
                needed.update(ids)
            out.append({"question_id": row["question_id"], "question": row["question"],
                        "query": value.get("query"), "route": value.get("route"),
                        "id_kind": "passage" if passage_mode else "entity",
                        "ranked_ids": ids, "matched_ids": value[matched],
                        "target_ids": (row.get("positive_passage_ids") if passage_mode
                                       else row.get("positive_entity_ids")) or [],
                        "error": value.get("error")})
        strategies[name] = out

    first_lists = None
    for name in ("adaptive-4b", "adaptive-9b", "adaptive-12b"):
        out = []
        for row in runs[name]["rows"]:
            first, follow = row["first_pass_ids"], row["followup_ids"]
            value = row["variants"]["adaptive"]
            ranked = ([item["id"] for item in fuse_passages(
                [{"id": key} for key in first], [{"id": key} for key in follow])]
                      if follow else first[:20])
            if ranked[:6] != value["ranked_passage_ids"]:
                raise ValueError(f"Reconstructed ranking differs: {name}/{row['question_id']}")
            needed.update(ranked)
            out.append({"question_id": row["question_id"], "question": row["question"],
                        "initial_query": row["initial_query"],
                        "query": value.get("followup_query"), "search_again": value.get("search_again"),
                        "id_kind": "passage", "ranked_ids": ranked,
                        "matched_ids": value.get("matched_passage_ids", []),
                        "target_ids": next(case["positive_passage_ids"] for case in challenge["questions"]
                                           if case["id"] == row["question_id"]),
                        "error": value.get("error")})
        strategies[name] = out
        lists = [(row["question_id"], row["first_pass_ids"]) for row in runs[name]["rows"]]
        if first_lists is not None and first_lists != lists:
            raise ValueError("Adaptive runs have different raw rankings")
        first_lists = lists

    raw = []
    for row in runs["adaptive-4b"]["rows"]:
        value = row["variants"]["first_pass"]
        ranked = row["first_pass_ids"]
        if ranked[:6] != value["ranked_passage_ids"]:
            raise ValueError(f"Raw ranking differs: {row['question_id']}")
        needed.update(ranked)
        raw.append({"question_id": row["question_id"], "question": row["question"],
                    "initial_query": row["initial_query"], "id_kind": "passage",
                    "ranked_ids": ranked, "matched_ids": value.get("matched_passage_ids", []),
                    "target_ids": next(case["positive_passage_ids"] for case in challenge["questions"]
                                       if case["id"] == row["question_id"])})
    strategies["adaptive-raw"] = raw
    missing = needed - by_id.keys()
    if missing:
        raise ValueError(f"Saved passage IDs missing from verified catalog: {sorted(missing)[:5]}")
    passages = {key: {field: by_id[key].get(field) for field in
                     ("id", "title", "text", "uri", "kind", "source_id")}
                for key in sorted(needed)}
    snapshot = {"version": 1, "source_commit": catalog["source_commit"], "index_identity": identity,
                "provenance": {name: {"run_id": runs[name]["run_id"],
                                      "file_sha256": hashlib.sha256((RUNS / path).read_bytes()).hexdigest()}
                               for name, path in RUN_FILES.items()},
                "note": "Recorded identifiers and queries; passage text reconstructed from the verified pinned catalog. Contextual entity results are de-duplicated entities among the first six passages. Adaptive top-20 order is reproduced from saved ranks using reciprocal-rank fusion and checked against recorded top six.",
                "strategies": strategies, "passages": passages}
    OUTPUT.write_text(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {OUTPUT}: {len(strategies)} strategies, {len(passages)} passages")


if __name__ == "__main__":
    main()
