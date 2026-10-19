"""Create blinded, randomized rating sheets for paired EVA reply comparisons."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def read_scenarios(path: Path) -> list[dict[str, str]]:
    required = {"scenario_id", "prompt", "expected_emotion", "adapted_response", "standard_response", "consent"}
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV is missing columns: {', '.join(sorted(missing))}")
        rows = []
        seen = set()
        for line, item in enumerate(reader, start=2):
            if item["consent"].strip().lower() not in {"true", "yes", "1"}:
                raise ValueError(f"{path}:{line}: consent must be explicitly true.")
            scenario_id = item["scenario_id"].strip()
            if not scenario_id or scenario_id in seen:
                raise ValueError(f"{path}:{line}: scenario_id must be present and unique.")
            seen.add(scenario_id)
            if not all(item[field].strip() for field in ("prompt", "adapted_response", "standard_response")):
                raise ValueError(f"{path}:{line}: prompt and both responses are required.")
            rows.append({key: value.strip() for key, value in item.items() if key})
    if not rows:
        raise ValueError("Input CSV contains no scenarios.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Private consented scenario/response CSV.")
    parser.add_argument("--ratings", required=True, type=Path, help="Blind rating sheet output CSV.")
    parser.add_argument("--key", required=True, type=Path, help="Condition key output CSV; keep this from raters.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.ratings.resolve() == args.key.resolve():
        parser.error("--ratings and --key must be different files")

    rng = random.Random(args.seed)
    rating_rows, key_rows = [], []
    for scenario in read_scenarios(args.input.resolve()):
        pair = [("adapted", scenario["adapted_response"]), ("standard", scenario["standard_response"])]
        rng.shuffle(pair)
        for condition, response in pair:
            presentation_id = f"R{rng.getrandbits(48):012x}"
            rating_rows.append({
                "presentation_id": presentation_id,
                "scenario_id": scenario["scenario_id"],
                "prompt": scenario["prompt"],
                "expected_emotion": scenario["expected_emotion"],
                "response": response,
                "rater_id": "",
                "helpfulness_1_to_5": "",
                "tone_fit_1_to_5": "",
                "assumptions_1_to_5": "",
                "comments": "",
            })
            key_rows.append({"presentation_id": presentation_id, "scenario_id": scenario["scenario_id"], "condition": condition})

    args.ratings.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.key.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.ratings.resolve().open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rating_rows[0]))
        writer.writeheader()
        writer.writerows(rating_rows)
    with args.key.resolve().open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)
    print(f"Created {len(rating_rows)} randomized blind ratings for {len(key_rows) // 2} paired scenarios.")
    print(f"Give raters only: {args.ratings.resolve()}")
    print(f"Keep condition key private: {args.key.resolve()}")


if __name__ == "__main__":
    main()
