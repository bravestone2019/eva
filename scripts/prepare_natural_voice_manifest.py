"""Create a speaker-separated annotation template for private natural speech."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True, help="Folder with one pseudonymous subfolder per speaker.")
    parser.add_argument("--output", type=Path, required=True, help="Private CSV output path outside the repository.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if args.dev_fraction <= 0 or args.test_fraction <= 0 or args.dev_fraction + args.test_fraction >= 1:
        parser.error("dev/test fractions must be positive and sum to less than 1")

    root = args.audio_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Audio folder not found: {root}")
    clips = []
    for audio_path in sorted(root.rglob("*")):
        if not audio_path.is_file() or audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        relative_parts = audio_path.relative_to(root).parts
        if len(relative_parts) < 2:
            raise ValueError(
                f"Place each clip in a pseudonymous speaker subfolder; found {audio_path} directly under the root."
            )
        clips.append({
            "speaker_group_id": relative_parts[0],
            "audio_path": str(audio_path.resolve()),
        })
    if not clips:
        raise ValueError(f"No supported audio clips found in {root}")

    speakers = sorted({row["speaker_group_id"] for row in clips})
    if len(speakers) < 3:
        raise ValueError("At least three distinct speaker folders are needed for train/dev/test splits.")
    random.Random(args.seed).shuffle(speakers)
    test_count = max(1, round(len(speakers) * args.test_fraction))
    dev_count = max(1, round(len(speakers) * args.dev_fraction))
    if test_count + dev_count >= len(speakers):
        raise ValueError("Not enough speakers for the requested train/dev/test fractions.")
    test_speakers = set(speakers[:test_count])
    dev_speakers = set(speakers[test_count:test_count + dev_count])

    for row in clips:
        speaker = row["speaker_group_id"]
        row["split"] = "test" if speaker in test_speakers else "dev" if speaker in dev_speakers else "train"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "split", "speaker_group_id", "audio_path", "transcript", "language_group",
        "self_reported_emotion", "self_reported_intensity",
        "perceived_emotion", "accent_group", "noise_level", "consent",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for row in sorted(clips, key=lambda item: (item["split"], item["speaker_group_id"], item["audio_path"])):
            writer.writerow({
                **row,
                "transcript": "",
                "language_group": "",
                "self_reported_emotion": "",
                "self_reported_intensity": "",
                "perceived_emotion": "",
                "accent_group": "",
                "noise_level": "",
                "consent": "false",
            })

    print(f"Created {args.output.resolve()} with {len(clips)} clip rows from {len(speakers)} speakers.")
    print("Split speakers: " + ", ".join(
        f"{split}={sum(row['split'] == split for row in clips)} clips/{len({r['speaker_group_id'] for r in clips if r['split'] == split})} speakers"
        for split in ("train", "dev", "test")
    ))
    print("Fill self_reported_emotion with the speaker's own label; perceived_emotion is optional and is never used as the training target.")
    print("Fill language_group (for example english, hindi, or hinglish), accent_group, and noise_level only when relevant; use broad, non-identifying values.")
    print("Set consent=true only after explicit permission to use that recording for this purpose.")


if __name__ == "__main__":
    main()
