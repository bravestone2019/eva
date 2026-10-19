"""Evaluate EVA's configured speech recognizer on consented natural speech."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import WHISPER_BACKEND, WHISPER_MODEL_NAME  # noqa: E402
from models.model_manager import ModelManager  # noqa: E402
from engines.speech_engine import SpeechEngine  # noqa: E402
from train_whisper import normalize_text, word_error_rate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Private CSV with consent, transcript, language_group, and split.")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path; transcripts/audio are not written.")
    parser.add_argument("--split", default="test", choices=("dev", "test"), help="Score only one split; default test.")
    args = parser.parse_args()

    path = args.manifest.resolve()
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"audio_path", "transcript", "consent", "split", "speaker_group_id", "language_group"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        rows = []
        speaker_splits = {}
        for line, row in enumerate(reader, start=2):
            if row.get("consent", "").strip().lower() not in {"true", "yes", "1"}:
                raise ValueError(f"{path}:{line}: consent must be explicitly true.")
            split = row.get("split", "").strip().lower()
            if split not in {"train", "dev", "test"}:
                raise ValueError(f"{path}:{line}: split must be train, dev, or test.")
            speaker = row.get("speaker_group_id", "").strip()
            if not speaker:
                raise ValueError(f"{path}:{line}: speaker_group_id is required.")
            previous = speaker_splits.setdefault(speaker, split)
            if previous != split:
                raise ValueError(f"{path}:{line}: speaker {speaker!r} appears in multiple splits.")
            transcript = row.get("transcript", "").strip()
            language = row.get("language_group", "").strip().lower()
            if split == args.split:
                if not transcript:
                    raise ValueError(f"{path}:{line}: transcript is required for ASR evaluation.")
                if not language:
                    raise ValueError(f"{path}:{line}: language_group is required for breakdowns.")
                audio = Path(row.get("audio_path", "").strip())
                if not audio.is_absolute():
                    audio = (path.parent / audio).resolve()
                if not audio.is_file():
                    raise FileNotFoundError(f"{path}:{line}: audio file not found: {audio}")
                rows.append({"audio_path": audio, "transcript": transcript, "language_group": language, "speaker_group_id": speaker})
    if not rows:
        raise ValueError(f"No rows found for split {args.split!r}.")

    manager = ModelManager()
    manager.load_whisper()
    speech = SpeechEngine(manager)
    references, predictions = [], []
    per_language: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        waveform, sample_rate = sf.read(row["audio_path"], dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != 16_000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16_000)
        prediction = speech.transcribe(np.asarray(waveform, dtype=np.float32))
        reference = normalize_text(row["transcript"])
        prediction = normalize_text(prediction)
        references.append(reference)
        predictions.append(prediction)
        per_language[row["language_group"]].append((reference, prediction))
        if index % 10 == 0 or index == len(rows):
            print(f"Scored {index}/{len(rows)} clips")

    report = {
        "backend": WHISPER_BACKEND,
        "model": WHISPER_MODEL_NAME,
        "split": args.split,
        "examples": len(rows),
        "speaker_groups": len({row["speaker_group_id"] for row in rows}),
        "wer": word_error_rate(references, predictions),
        "wer_by_language_group": {
            language: {"examples": len(values), "wer": word_error_rate([v[0] for v in values], [v[1] for v in values])}
            for language, values in sorted(per_language.items())
        },
        "normalization": "Unicode NFC, lowercase, and basic punctuation removal.",
        "warning": "Use a speaker-held-out test set. This small consented set estimates only its sampled speakers, language mix, recording conditions, and transcription conventions.",
        "privacy": "Report excludes audio paths, transcripts, and predictions.",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved to {output}")


if __name__ == "__main__":
    main()
