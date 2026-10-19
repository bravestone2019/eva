"""Evaluate EVA's affect predictions on a consented, speaker-separated CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines.affect_engine import AUDIO_LABELS, AffectEngine  # noqa: E402


EMOTIONS = ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised")
THRESHOLDS = tuple(round(0.35 + index * 0.05, 2) for index in range(10))


def predict(result: dict, threshold: float) -> str:
    if result.get("disagreement"):
        return "uncertain"
    scores = result.get("scores") or {}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return "uncertain"
    label, confidence = ranked[0]
    margin = confidence - (ranked[1][1] if len(ranked) > 1 else 0.0)
    if confidence < threshold or margin < 0.08:
        return "uncertain"
    return label


def macro_f1(rows: list[dict], threshold: float) -> float:
    values = []
    labels_present = sorted({row["expected"] for row in rows})
    for label in labels_present:
        true_positive = sum(row["expected"] == label and predict(row["result"], threshold) == label for row in rows)
        false_positive = sum(row["expected"] != label and predict(row["result"], threshold) == label for row in rows)
        false_negative = sum(row["expected"] == label and predict(row["result"], threshold) != label for row in rows)
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(values) / len(values)


def summarize(rows: list[dict], threshold: float) -> dict:
    if not rows:
        return {"count": 0}
    predictions = [predict(row["result"], threshold) for row in rows]
    correct = sum(prediction == row["expected"] for row, prediction in zip(rows, predictions))
    covered = [(row, prediction) for row, prediction in zip(rows, predictions) if prediction != "uncertain"]
    labels = sorted({row["expected"] for row in rows})
    per_class = {}
    for label in labels:
        tp = sum(row["expected"] == label and prediction == label for row, prediction in zip(rows, predictions))
        fp = sum(row["expected"] != label and prediction == label for row, prediction in zip(rows, predictions))
        fn = sum(row["expected"] == label and prediction != label for row, prediction in zip(rows, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {
            "support": sum(row["expected"] == label for row in rows),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        }
    return {
        "count": len(rows),
        "accuracy": round(correct / len(rows), 4),
        "macro_f1": round(macro_f1(rows, threshold), 4),
        "coverage": round(len(covered) / len(rows), 4),
        "accuracy_when_confident": round(
            sum(row["expected"] == prediction for row, prediction in covered) / len(covered), 4
        ) if covered else None,
        "uncertain_count": len(rows) - len(covered),
        "per_class": per_class,
    }


def grouped_metrics(rows: list[dict], field: str, threshold: float) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        value = row.get(field, "")
        if value:
            groups[value].append(row)
    return {
        key: summarize(group, threshold)
        for key, group in sorted(groups.items())
    }


def brier_score(rows: list[dict]) -> float | None:
    if not rows:
        return None
    totals = []
    for row in rows:
        scores = row["result"].get("scores") or {}
        denominator = sum(max(float(scores.get(label, 0.0)), 0.0) for label in EMOTIONS)
        if denominator <= 0:
            probabilities = {label: 0.0 for label in EMOTIONS}
        else:
            probabilities = {label: max(float(scores.get(label, 0.0)), 0.0) / denominator for label in EMOTIONS}
        totals.append(sum(
            (probabilities[label] - float(row["expected"] == label)) ** 2
            for label in EMOTIONS
        ))
    return round(sum(totals) / len(totals), 4)


def evaluate_manifest(path: Path) -> dict:
    engine = AffectEngine()
    rows = []
    speaker_splits: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        for line_number, item in enumerate(csv.DictReader(source), start=2):
            if item.get("consent", "").strip().lower() not in {"true", "yes", "1"}:
                raise ValueError(f"{path}:{line_number}: consent must be explicitly true.")
            raw_expected = (
                item.get("self_reported_emotion", "").strip().lower()
                or item.get("expected_emotion", "").strip().lower()
            )
            expected = AUDIO_LABELS.get(raw_expected, raw_expected)
            if expected not in EMOTIONS:
                raise ValueError(f"{path}:{line_number}: expected_emotion must be one of {', '.join(EMOTIONS)}.")
            split = item.get("split", "test").strip().lower() or "test"
            if split not in {"train", "dev", "test"}:
                raise ValueError(f"{path}:{line_number}: split must be train, dev, or test.")
            speaker_group_id = item.get("speaker_group_id", "").strip()
            if speaker_group_id:
                previous_split = speaker_splits.setdefault(speaker_group_id, split)
                if previous_split != split:
                    raise ValueError(
                        f"{path}:{line_number}: speaker_group_id {speaker_group_id!r} appears in "
                        f"both {previous_split!r} and {split!r} splits."
                    )
            audio_path = Path(item.get("audio_path", "").strip())
            if not audio_path.is_absolute():
                audio_path = (path.parent / audio_path).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"{path}:{line_number}: audio file not found: {audio_path}")
            result = engine.analyze_file(item.get("transcript", ""), str(audio_path))
            rows.append({
                "expected": expected,
                "result": result,
                "split": split,
                "speaker_group_id": speaker_group_id,
                "accent_group": item.get("accent_group", "").strip(),
                "noise_level": item.get("noise_level", "").strip(),
                "language_group": item.get("language_group", "").strip(),
            })

    if not rows:
        raise ValueError(f"{path}: manifest contains no examples.")

    development = [row for row in rows if row["split"] == "dev"]
    test = [row for row in rows if row["split"] == "test"]
    calibration = development or rows
    threshold = max(
        THRESHOLDS,
        key=lambda candidate: (macro_f1(calibration, candidate), summarize(calibration, candidate)["coverage"]),
    )
    used_fallback_test = not test
    test = test or rows
    warnings = []
    if not development:
        warnings.append("No dev split: confidence threshold was selected on all rows; results are exploratory.")
    if used_fallback_test:
        warnings.append("No test split: metrics below are not held-out test results.")
    if len({row["speaker_group_id"] for row in rows if row["speaker_group_id"]}) < 3:
        warnings.append("Fewer than three pseudonymous speaker groups are present; speaker generalization is uncertain.")
    return {
        "manifest": path.name,
        "labels_in_manifest": sorted({row["expected"] for row in rows}),
        "threshold_selected_on": "dev" if development else "all rows (no dev split provided)",
        "selected_confidence_threshold": threshold,
        "dev": summarize(development, threshold) if development else None,
        "test": summarize(test, threshold),
        "test_brier_score": brier_score(test),
        "test_by_speaker_group": grouped_metrics(test, "speaker_group_id", threshold),
        "test_by_accent_group": grouped_metrics(test, "accent_group", threshold),
        "test_by_noise_level": grouped_metrics(test, "noise_level", threshold),
        "test_by_language_group": grouped_metrics(test, "language_group", threshold),
        "speaker_groups_by_split": {
            split: len({row["speaker_group_id"] for row in rows if row["split"] == split and row["speaker_group_id"]})
            for split in ("train", "dev", "test")
        },
        "warnings": warnings,
        "warning": "Use a separate speaker-held-out test split. Tune thresholds on dev only; metrics estimate expressed affect and do not establish a person's internal emotional state.",
    }


def summarize_feedback(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    ratings = data.get("ratings", [])
    groups: dict[str, list[dict]] = defaultdict(list)
    styles: dict[str, list[dict]] = defaultdict(list)
    reasons: dict[str, int] = defaultdict(int)
    for rating in ratings:
        bucket = "adapted" if rating.get("adapted") else "standard_or_uncertain"
        groups[bucket].append(rating)
        styles[str(rating.get("response_style", "standard"))].append(rating)
        if rating.get("reason"):
            reasons[str(rating["reason"])] += 1
    by_group = {}
    def summarize_group(group: list[dict]) -> dict:
        latencies = [float(item["latency_ms"]) for item in group if item.get("latency_ms")]
        helpful_count = sum(item.get("rating") == "helpful" for item in group)
        return {
            "count": len(group),
            "helpful_rate": round(helpful_count / len(group), 4) if group else None,
            "mean_response_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        }
    for label, group in sorted(groups.items()):
        by_group[label] = summarize_group(group)
    return {
        "rating_groups": by_group,
        "rating_groups_by_response_style": {
            label: summarize_group(group) for label, group in sorted(styles.items())
        },
        "not_helpful_reasons": dict(sorted(reasons.items())),
        "interpretation": "Style-group comparisons are observational, not a randomized paired experiment.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="CSV with explicitly consented, labeled voice examples")
    parser.add_argument("--feedback", type=Path, help="JSON exported from EVA's reply-rating controls")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report")
    args = parser.parse_args()
    if not args.manifest and not args.feedback:
        parser.error("Provide --manifest, --feedback, or both.")

    report = {}
    if args.manifest:
        report["affect_model"] = evaluate_manifest(args.manifest.resolve())
    if args.feedback:
        report["reply_feedback"] = summarize_feedback(args.feedback.resolve())
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
