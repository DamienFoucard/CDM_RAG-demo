#!/usr/bin/env python3
"""Compare two paired benchmark runs; bootstrap varies cases, not people."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def paired_comparison(baseline: dict, candidate: dict, variant: str, samples: int = 10000) -> dict:
    for key in ("dataset_sha256", "source_commit", "split"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"runs differ in {key}")
    left = {case["id"]: case for case in baseline["cases"]}
    right = {case["id"]: case for case in candidate["cases"]}
    if set(left) != set(right) or not left:
        raise ValueError("runs must contain the same nonempty question set")
    for case_id in left:
        if (left[case_id]["question"], left[case_id]["positive_entity_ids"]) != (
                right[case_id]["question"], right[case_id]["positive_entity_ids"]):
            raise ValueError("paired cases do not have identical questions and labels")
    deltas = [right[case_id]["variants"][variant]["mrr_at_6"] -
              left[case_id]["variants"][variant]["mrr_at_6"] for case_id in sorted(left)]
    rng = random.Random(20261122)
    bootstrap = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(samples))
    return {"dataset_sha256": baseline["dataset_sha256"], "split": baseline["split"],
            "variant": variant, "baseline": baseline["embedding_model"], "candidate": candidate["embedding_model"],
            "case_count": len(deltas), "mrr_delta": round(statistics.mean(deltas), 4),
            "paired_bootstrap_95_percent": [round(bootstrap[int(samples * .025)], 4),
                                             round(bootstrap[min(samples - 1, int(samples * .975))], 4)],
            "wins": sum(delta > 0 for delta in deltas), "ties": sum(delta == 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
            "limitation": "Resampling this synthetic, unreviewed question set does not establish uncertainty for a real user population."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--variant", default="hybrid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = paired_comparison(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()), args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
