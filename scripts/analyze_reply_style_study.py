"""Analyze ratings from the blinded paired reply-style study."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SCORES = ("helpfulness_1_to_5", "tone_fit_1_to_5", "assumptions_1_to_5")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", required=True, type=Path, help="Completed blind rating CSV.")
    parser.add_argument("--key", required=True, type=Path, help="Private condition key CSV from preparation.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ratings = read_csv(args.ratings.resolve())
    key_rows = read_csv(args.key.resolve())
    keys = {row["presentation_id"].strip(): row for row in key_rows if row.get("presentation_id", "").strip()}
    if len(keys) != len(key_rows):
        raise ValueError("Condition key contains blank or duplicate presentation IDs.")
    if not keys:
        raise ValueError("Condition key has no presentation IDs.")

    by_pair: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    by_emotion: dict[str, list[dict]] = defaultdict(list)
    seen_ratings = set()
    for row in ratings:
        presentation_id = row.get("presentation_id", "").strip()
        if presentation_id not in keys:
            raise ValueError(f"Unknown presentation ID in ratings: {presentation_id!r}.")
        key = keys[presentation_id]
        rater_id = row.get("rater_id", "").strip()
        if not rater_id:
            raise ValueError(f"{presentation_id}: rater_id is required; use a pseudonym such as rater_01.")
        if (presentation_id, rater_id) in seen_ratings:
            raise ValueError(f"Duplicate rating from {rater_id!r} for presentation {presentation_id}.")
        seen_ratings.add((presentation_id, rater_id))
        if row.get("scenario_id", "").strip() != key.get("scenario_id", "").strip():
            raise ValueError(f"Scenario ID mismatch for presentation {presentation_id}.")
        condition = key.get("condition", "").strip()
        if condition not in {"adapted", "standard"}:
            raise ValueError(f"Invalid condition for presentation {presentation_id}.")
        scores = {}
        for field in SCORES:
            try:
                score = int(row.get(field, ""))
            except ValueError as exc:
                raise ValueError(f"{presentation_id}: {field} must be filled with an integer 1-5.") from exc
            if not 1 <= score <= 5:
                raise ValueError(f"{presentation_id}: {field} must be between 1 and 5.")
            scores[field] = score
        scenario_id = key["scenario_id"].strip()
        pair_key = (scenario_id, row.get("expected_emotion", "").strip(), rater_id)
        if condition in by_pair[pair_key]:
            raise ValueError(f"Duplicate {condition} rating for scenario {scenario_id} by {rater_id}.")
        record = {"condition": condition, **scores}
        by_pair[pair_key][condition] = record
        by_emotion[row.get("expected_emotion", "unspecified").strip() or "unspecified"].append(record)

    pairs = [pair for pair in by_pair.values() if set(pair) == {"adapted", "standard"}]
    raters = {row.get("rater_id", "").strip() for row in ratings}
    expected_ratings = {(presentation_id, rater) for presentation_id in keys for rater in raters}
    if len(pairs) != len(by_pair) or seen_ratings != expected_ratings:
        raise ValueError("Every scenario must have one adapted and one standard rating.")

    def compare(pair_rows: list[dict]) -> dict:
        report = {}
        for field in SCORES:
            adapted = [pair["adapted"][field] for pair in pair_rows]
            standard = [pair["standard"][field] for pair in pair_rows]
            # Lower assumption scores mean fewer unwarranted assumptions, so this metric is reversed.
            delta = [b - a if field == "assumptions_1_to_5" else a - b for a, b in zip(adapted, standard)]
            report[field] = {
                "adapted_mean": round(sum(adapted) / len(adapted), 3) if adapted else None,
                "standard_mean": round(sum(standard) / len(standard), 3) if standard else None,
                "adapted_preferred": sum(value > 0 for value in delta),
                "standard_preferred": sum(value < 0 for value in delta),
                "ties": sum(value == 0 for value in delta),
                "mean_paired_delta_toward_adapted": round(sum(delta) / len(delta), 3) if delta else None,
            }
        return report

    report = {
        "unique_scenarios": len({scenario for scenario, _, _ in by_pair}),
        "paired_ratings": len(pairs),
        "raters": len(raters),
        "overall": compare(pairs),
        "by_expected_emotion": {
            emotion: compare([pair for (scenario, pair_emotion, rater), pair in by_pair.items() if pair_emotion == emotion and set(pair) == {"adapted", "standard"}])
            for emotion in sorted(by_emotion)
        },
        "interpretation": "Descriptive paired ratings only; small or convenience samples do not establish broad effectiveness or statistical significance.",
        "assumptions_scale": "1 means few unwarranted assumptions, 5 means many; this metric is reversed when calculating the paired delta.",
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
