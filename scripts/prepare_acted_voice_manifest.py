"""Prepare a speaker-separated eight-class manifest from local RAVDESS and CREMA-D WAVs."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path


RAVDESS_LABELS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
CREMAD_LABELS = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
RAVDESS_BASIS = (
    "RAVDESS CC BY-NC-SA 4.0; comply with attribution, non-commercial, "
    "and share-alike terms"
)
CREMAD_BASIS = "CREMA-D ODbL 1.0 and DbCL 1.0; comply with attribution and license terms"


def discover_ravdess(root: Path) -> list[dict]:
    if not root.is_dir():
        raise FileNotFoundError(f"RAVDESS folder not found: {root}")
    rows = []
    for audio_path in root.rglob("*.wav"):
        parts = audio_path.stem.split("-")
        # RAVDESS: audio-only speech; include all eight original emotion labels.
        if len(parts) != 7 or parts[0] != "03" or parts[1] != "01":
            continue
        label = RAVDESS_LABELS.get(parts[2])
        if not label:
            continue
        rows.append({
            "speaker_group_id": f"ravdess_actor_{parts[6]}",
            "audio_path": str(audio_path.resolve()),
            "expected_emotion": label,
            "source_dataset": "RAVDESS",
            "source_license": "CC BY-NC-SA 4.0",
            "permission_basis": RAVDESS_BASIS,
        })
    if not rows:
        raise ValueError(f"No supported RAVDESS speech WAVs found under {root}")
    return rows


def discover_cremad(root: Path) -> list[dict]:
    if not root.is_dir():
        raise FileNotFoundError(f"CREMA-D AudioWAV folder not found: {root}")
    rows = []
    for audio_path in root.glob("*.wav"):
        parts = audio_path.stem.split("_")
        if len(parts) != 4:
            continue
        label = CREMAD_LABELS.get(parts[2].upper())
        if not label:
            continue
        rows.append({
            "speaker_group_id": f"cremad_actor_{parts[0]}",
            "audio_path": str(audio_path.resolve()),
            "expected_emotion": label,
            "source_dataset": "CREMA-D",
            "source_license": "ODbL 1.0; contents under DbCL 1.0",
            "permission_basis": CREMAD_BASIS,
        })
    if not rows:
        raise ValueError(f"No supported CREMA-D WAVs found under {root}")
    return rows


def assign_speaker_splits(rows: list[dict], seed: int, dev_fraction: float, test_fraction: float):
    speakers_by_source: dict[str, set[str]] = {}
    for row in rows:
        source = row["source_dataset"]
        speakers_by_source.setdefault(source, set()).add(row["speaker_group_id"])

    split_by_speaker = {}
    for source, speakers in speakers_by_source.items():
        ordered = sorted(speakers)
        random.Random(f"{seed}:{source}").shuffle(ordered)
        test_count = max(1, round(len(ordered) * test_fraction))
        dev_count = max(1, round(len(ordered) * dev_fraction))
        if test_count + dev_count >= len(ordered):
            raise ValueError(f"Not enough {source} speakers for train/dev/test splits.")
        test_speakers = set(ordered[:test_count])
        dev_speakers = set(ordered[test_count:test_count + dev_count])
        for speaker in ordered:
            split_by_speaker[speaker] = (
                "test" if speaker in test_speakers
                else "dev" if speaker in dev_speakers
                else "train"
            )

    for row in rows:
        row["split"] = split_by_speaker[row["speaker_group_id"]]
    return split_by_speaker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ravdess-dir", type=Path, required=True, help="Root containing Actor_XX folders.")
    parser.add_argument("--cremad-dir", type=Path, required=True, help="CREMA-D AudioWAV folder.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path; keep it outside Git.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if args.dev_fraction <= 0 or args.test_fraction <= 0 or args.dev_fraction + args.test_fraction >= 1:
        parser.error("dev/test fractions must be positive and sum to less than 1")

    rows = discover_ravdess(args.ravdess_dir.resolve()) + discover_cremad(args.cremad_dir.resolve())
    split_by_speaker = assign_speaker_splits(
        rows, args.seed, args.dev_fraction, args.test_fraction
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "split", "speaker_group_id", "audio_path", "expected_emotion",
        "consent", "permission_basis", "source_dataset", "source_license",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["split"], item["source_dataset"], item["speaker_group_id"], item["audio_path"])):
            writer.writerow({**row, "consent": "false"})

    print(f"Saved {len(rows)} supported audio clips to {args.output.resolve()}")
    print(f"Speaker groups: {len(split_by_speaker)} (split by actor, separately per dataset)")
    for split in ("train", "dev", "test"):
        subset = [row for row in rows if row["split"] == split]
        print(f"{split}: {len(subset)} clips; labels={dict(sorted(Counter(r['expected_emotion'] for r in subset).items()))}")
    print("Canonical labels: neutral, calm, happy, sad, angry, fearful, disgust, surprised")
    print("RAVDESS supplies all eight labels; CREMA-D supplies angry, disgust, fearful, happy, neutral, and sad.")
    print("Review each source license before using or distributing trained weights.")


if __name__ == "__main__":
    main()
