#!/usr/bin/env python3
"""Flag cases where an automated judge disagrees with example positive labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(result: dict) -> dict:
    disagreements = []
    total = 0
    for case in result["cases"]:
        for entity_id in case["positive_entity_ids"]:
            decision = case["judgments"].get(entity_id)
            if decision is None:
                raise ValueError("a labeled entity is missing from the judged pool")
            total += 1
            if decision["grade"] != 2:
                disagreements.append({"question_id": case["id"], "question": case["question"],
                                      "entity_id": entity_id, "judge_grade": decision["grade"],
                                      "judge_reason": decision["reason"]})
    if not total:
        raise ValueError("no labeled positives to audit")
    return {"dataset_sha256": result["dataset_sha256"], "split": result["split"],
            "judge_model": result["judge_model"], "judge_model_digest": result["judge_model_digest"],
            "labeled_positives": total, "judge_direct_count": total - len(disagreements),
            "anchor_disagreements": disagreements,
            "limitation": "Example positives and judge decisions are both unreviewed; disagreements need human adjudication."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("judge_result", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = audit(json.loads(args.judge_result.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "labeled_positives": output["labeled_positives"],
                      "disagreements": len(output["anchor_disagreements"])}, indent=2))


if __name__ == "__main__":
    main()
