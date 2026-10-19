"""Evaluate EVA's live text-affect path on a consented human-reviewed CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engines.affect_engine import AffectEngine  # noqa: E402
from scripts.evaluate_affect import EMOTIONS, THRESHOLDS, macro_f1, predict, summarize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Human-reviewed CSV based on evaluation/text_affect_review.example.csv.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.manifest.resolve()
    required = {"split", "conversation_group_id", "language_group", "text", "human_emotion", "consent"}
    records = []
    group_splits = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            if row.get("consent", "").strip().lower() not in {"true", "yes", "1"}:
                raise ValueError(f"{path}:{line}: consent must be explicitly true.")
            split = row.get("split", "").strip().lower()
            if split not in {"train", "dev", "test"}:
                raise ValueError(f"{path}:{line}: split must be train, dev, or test.")
            group = row.get("conversation_group_id", "").strip()
            if not group:
                raise ValueError(f"{path}:{line}: conversation_group_id is required to prevent related-text leakage.")
            previous = group_splits.setdefault(group, split)
            if previous != split:
                raise ValueError(f"{path}:{line}: conversation group {group!r} appears in multiple splits.")
            label = row.get("human_emotion", "").strip().lower()
            if label not in EMOTIONS:
                raise ValueError(f"{path}:{line}: human_emotion must be one of {', '.join(EMOTIONS)}.")
            text = row.get("text", "").strip()
            language = row.get("language_group", "").strip().lower()
            if not text or not language:
                raise ValueError(f"{path}:{line}: text and language_group are required.")
            records.append({"split": split, "group": group, "language": language, "expected": label, "text": text})
    if not records:
        raise ValueError("Manifest contains no annotated examples.")

    engine = AffectEngine()
    evaluated = []
    for index, row in enumerate(records, start=1):
        result = engine.analyze(text=row["text"])
        evaluated.append({**row, "result": result})
        if index % 25 == 0 or index == len(records):
            print(f"Scored {index}/{len(records)} texts")
    dev = [row for row in evaluated if row["split"] == "dev"]
    test = [row for row in evaluated if row["split"] == "test"]
    calibration = dev or evaluated
    threshold = max(THRESHOLDS, key=lambda value: (macro_f1(calibration, value), summarize(calibration, value)["coverage"]))
    if not test:
        test = evaluated
    report = {
        "model": "EVA configured live text affect model",
        "labels": sorted({row["expected"] for row in evaluated}),
        "selected_confidence_threshold": threshold,
        "threshold_selected_on": "dev" if dev else "all rows (exploratory only)",
        "dev": summarize(dev, threshold) if dev else None,
        "test": summarize(test, threshold),
        "test_by_language_group": {
            language: summarize([row for row in test if row["language"] == language], threshold)
            for language in sorted({row["language"] for row in test})
        },
        "conversation_groups_by_split": {
            split: len({row["group"] for row in evaluated if row["split"] == split})
            for split in ("train", "dev", "test")
        },
        "warnings": [
            "Threshold selection on all rows is exploratory because there is no dev split." if not dev else None,
            "No held-out test split; reported test metrics are exploratory." if not any(row["split"] == "test" for row in evaluated) else None,
            "Text affect reflects cues in the utterance, not a diagnosis or certain knowledge of internal emotion.",
        ],
        "privacy": "Report excludes reviewed text and model predictions.",
    }
    report["warnings"] = [warning for warning in report["warnings"] if warning]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved to {output}")


if __name__ == "__main__":
    main()
