"""Fine-tune/evaluate a Hugging Face Whisper checkpoint on a prepared FLEURS split."""

from __future__ import annotations

import argparse
import io
import json
import re
import unicodedata
from pathlib import Path

import torch


class WhisperDataCollator:
    """Decode audio and build model inputs only for the current batch.

    Keeping the original audio in the Dataset avoids a large, precomputed
    feature cache and sidesteps the Windows multiprocessing path in
    ``Dataset.map``.
    """

    def __init__(self, processor, librosa, soundfile):
        self.processor = processor
        self.librosa = librosa
        self.soundfile = soundfile

    def __call__(self, features):
        audio_features = []
        label_features = []
        for item in features:
            audio = item["audio"]
            audio_bytes = audio.get("bytes")
            if audio_bytes is None:
                raise ValueError(
                    f"FLEURS audio bytes are missing for {audio.get('path')!r}; "
                    "prepare the dataset with embedded audio files."
                )
            waveform, sampling_rate = self.soundfile.read(
                io.BytesIO(audio_bytes), dtype="float32"
            )
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if sampling_rate != 16_000:
                waveform = self.librosa.resample(
                    waveform, orig_sr=sampling_rate, target_sr=16_000
                )
            input_features = self.processor.feature_extractor(
                waveform, sampling_rate=16_000
            ).input_features[0]
            audio_features.append({"input_features": input_features})
            label_features.append({
                "input_ids": self.processor.tokenizer(
                    item["transcription"], max_length=448, truncation=True
                ).input_ids
            })

        batch = self.processor.feature_extractor.pad(audio_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if labels.shape[1] > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def normalize_text(text):
    text = unicodedata.normalize("NFC", text).lower().strip()
    return re.sub(r"[.,!?;:\"'()\[\]{}]", "", text)


def word_error_rate(references, predictions):
    """Compute corpus WER without downloading a metric implementation."""
    total_errors = 0
    total_reference_words = 0
    for reference, prediction in zip(references, predictions):
        reference_words = reference.split()
        prediction_words = prediction.split()
        previous = list(range(len(prediction_words) + 1))
        for ref_index, ref_word in enumerate(reference_words, start=1):
            current = [ref_index]
            for pred_index, pred_word in enumerate(prediction_words, start=1):
                current.append(min(
                    previous[pred_index] + 1,
                    current[pred_index - 1] + 1,
                    previous[pred_index - 1] + (ref_word != pred_word),
                ))
            previous = current
        total_errors += previous[-1]
        total_reference_words += len(reference_words)
    return total_errors / max(total_reference_words, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Prepared DatasetDict with train/validation/test.")
    parser.add_argument("--language", choices=["english", "hindi"], required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--eval-only", action="store_true", help="Score the pretrained checkpoint on the test split without training.")
    args = parser.parse_args()

    try:
        from datasets import Audio, load_from_disk
        import librosa
        import soundfile as sf
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
    except ImportError as error:
        raise SystemExit("Install optional training dependencies first: pip install -r requirements-training.txt") from error

    language_name = "English" if args.language == "english" else "Hindi"
    processor = WhisperProcessor.from_pretrained(args.model, language=language_name, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = language_name
    model.generation_config.task = "transcribe"
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language_name, task="transcribe")
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.forced_decoder_ids = None

    datasets = load_from_disk(str(args.dataset_dir.resolve()))
    for split in ("train", "validation", "test"):
        if split not in datasets:
            raise ValueError(f"Dataset is missing the {split!r} split.")
        # `decode=False` avoids the torchcodec requirement added by newer
        # versions of `datasets`; soundfile can decode the FLEURS bytes here.
        datasets[split] = datasets[split].cast_column(
            "audio", Audio(sampling_rate=16_000, decode=False)
        )

    def compute_metrics(prediction):
        pred_ids = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        label_ids = prediction.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predictions = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        references = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {
            "wer": word_error_rate(
                [normalize_text(value) for value in references],
                [normalize_text(value) for value in predictions],
            )
        }

    output_dir = args.output_dir.resolve()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        num_train_epochs=args.epochs,
        # Gradient checkpointing currently triggers a repeated-backward error
        # with Whisper under the project's Torch/Transformers combination.
        # Keep it disabled; the small-model training path uses batch size 1.
        gradient_checkpointing=False,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=not args.eval_only,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=42,
        data_seed=42,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None if args.eval_only else datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=WhisperDataCollator(processor, librosa, sf),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    if not args.eval_only:
        trainer.train()
        trainer.save_model(str(output_dir))
        processor.save_pretrained(str(output_dir))

    result = trainer.predict(datasets["test"], metric_key_prefix="test")
    report = {
        "language": args.language,
        "base_model": args.model,
        "model_evaluated": args.model if args.eval_only else str(output_dir),
        "train_rows": len(datasets["train"]),
        "validation_rows": len(datasets["validation"]),
        "test_rows": len(datasets["test"]),
        "test_wer": float(result.metrics.get("test_wer", float("nan"))),
        "normalization": "Unicode NFC, lowercase, and basic punctuation removal; apply the same normalization to all comparisons.",
        "warning": "Do not use the test split for checkpoint or hyperparameter selection.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        report_name = "baseline-report.json" if args.model == "openai/whisper-small" else "eval-report.json"
    else:
        report_name = "test-report.json"
    report_path = output_dir / report_name
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
