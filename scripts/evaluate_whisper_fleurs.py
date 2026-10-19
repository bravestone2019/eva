"""Evaluate EVA's OpenAI Whisper runtime checkpoint on prepared FLEURS tests."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch
import whisper

from train_whisper import normalize_text, word_error_rate


LANGUAGES = {
    "english": "en",
    "hindi": "hi",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/fleurs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fleurs-current-whisper"))
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES))
    args = parser.parse_args()

    try:
        import librosa
        import soundfile as sf
        from datasets import Audio, load_from_disk
    except ImportError as error:
        raise SystemExit(
            "Install EVA's runtime and dataset dependencies before evaluation."
        ) from error

    model = whisper.load_model(args.model)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for language in args.languages:
        dataset_path = args.dataset_root / f"{language}-small"
        datasets = load_from_disk(str(dataset_path.resolve()))
        test = datasets["test"].cast_column(
            "audio", Audio(sampling_rate=16_000, decode=False)
        )
        references = []
        predictions = []
        for index, row in enumerate(test):
            audio = row["audio"]
            audio_bytes = audio.get("bytes")
            if audio_bytes is None:
                raise ValueError(
                    f"Audio bytes missing for {audio.get('path')!r}; "
                    "the prepared split must include embedded audio."
                )
            waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if sample_rate != 16_000:
                waveform = librosa.resample(
                    waveform, orig_sr=sample_rate, target_sr=16_000
                )
            result = model.transcribe(
                waveform,
                language=LANGUAGES[language],
                task="transcribe",
                fp16=torch.cuda.is_available(),
                verbose=False,
            )
            references.append(normalize_text(row["transcription"]))
            predictions.append(normalize_text(result["text"]))
            if (index + 1) % 25 == 0 or index + 1 == len(test):
                print(f"{language}: {index + 1}/{len(test)}")

        report = {
            "language": language,
            "model": args.model,
            "runtime_default": "EVA_WHISPER_BACKEND=openai; EVA_WHISPER_MODEL=large-v3-turbo",
            "test_examples": len(test),
            "test_wer": word_error_rate(references, predictions),
            "normalization": "Unicode NFC, lowercase, and basic punctuation removal; same normalization as the Transformers Whisper evaluator.",
            "warning": "FLEURS is read speech; this does not establish spontaneous microphone performance.",
        }
        report_path = output_dir / f"{language}-report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
