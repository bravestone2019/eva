"""Evaluate or fine-tune EVA's eight-class voice emotion model from a private CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
    __version__ as TRANSFORMERS_VERSION,
)


LABELS = ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
SOURCE_LABELS = {
    "ang": "angry", "anger": "angry", "angry": "angry", "frustrated": "angry",
    "dis": "disgust", "disgust": "disgust", "disgusted": "disgust",
    "fea": "fearful", "fear": "fearful", "fearful": "fearful",
    "hap": "happy", "happy": "happy", "happiness": "happy",
    "excited": "happy", "excitement": "happy", "joy": "happy",
    "cal": "calm", "calm": "calm",
    "sad": "sad", "sadness": "sad",
    "neu": "neutral", "neutral": "neutral",
    "sur": "surprised", "surprise": "surprised", "surprised": "surprised",
}
SAMPLE_RATE = 16_000


def load_manifest(path: Path) -> dict[str, list[dict]]:
    rows_by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    speakers: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"split", "speaker_group_id", "audio_path", "expected_emotion"}
        label_column = "self_reported_emotion" if "self_reported_emotion" in (reader.fieldnames or []) else "expected_emotion"
        required.discard("expected_emotion")
        required.add(label_column)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        for line, item in enumerate(reader, start=2):
            consented = item.get("consent", "").strip().lower() in {"true", "yes", "1"}
            permission_basis = item.get("permission_basis", "").strip()
            if not consented and not permission_basis:
                raise ValueError(
                    f"{path}:{line}: provide consent=true for consented recordings, "
                    "or a permission_basis for licensed corpus data."
                )
            raw_label = item.get(label_column, "").strip().lower()
            label = SOURCE_LABELS.get(raw_label, raw_label)
            if label not in LABELS:
                raise ValueError(
                    f"{path}:{line}: {label_column} must be one of {', '.join(LABELS)}. "
                    "Natural-speech training uses the speaker's self-reported emotion."
                )
            split = item.get("split", "").strip().lower()
            if split == "validation":
                split = "dev"
            if split not in rows_by_split:
                raise ValueError(f"{path}:{line}: split must be train, dev, or test.")
            speaker = item.get("speaker_group_id", "").strip()
            if not speaker:
                raise ValueError(f"{path}:{line}: speaker_group_id is required.")
            previous_split = speakers.setdefault(speaker, split)
            if previous_split != split:
                raise ValueError(f"Speaker {speaker!r} appears in both {previous_split!r} and {split!r} splits.")
            audio_path = Path(item.get("audio_path", "").strip())
            if not audio_path.is_absolute():
                audio_path = (path.parent / audio_path).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"{path}:{line}: audio file not found: {audio_path}")
            rows_by_split[split].append({
                "audio_path": audio_path,
                "label": label,
                "speaker_group_id": speaker,
                "source_dataset": item.get("source_dataset", "user-provided").strip(),
                "source_license": item.get("source_license", "").strip(),
                "perceived_emotion": item.get("perceived_emotion", "").strip().lower(),
            })

    if not rows_by_split["train"]:
        raise ValueError("The manifest needs at least one train row.")
    missing_train_labels = set(LABELS) - {row["label"] for row in rows_by_split["train"]}
    if missing_train_labels:
        raise ValueError(f"Train split is missing classes: {', '.join(sorted(missing_train_labels))}.")
    for split in ("dev", "test"):
        if not rows_by_split[split]:
            raise ValueError(f"The manifest needs a speaker-separated {split} split.")
    return rows_by_split


def model_output_labels(model) -> dict[int, str]:
    raw_labels = model.config.id2label
    id_to_eva = {}
    for index in range(model.config.num_labels):
        raw = str(raw_labels.get(index, raw_labels.get(str(index), ""))).strip().lower()
        mapped = SOURCE_LABELS.get(raw)
        if not mapped:
            raise ValueError(
                f"Model output {index} is {raw!r}, which is not in the supported voice-label map."
            )
        id_to_eva[index] = mapped
    return id_to_eva


class VoiceDataset(Dataset):
    def __init__(self, rows: list[dict], label_to_id: dict[str, int], extractor):
        self.rows = rows
        self.label_to_id = label_to_id
        self.extractor = extractor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import librosa

        row = self.rows[index]
        waveform, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.size < SAMPLE_RATE // 2 or not np.isfinite(waveform).all():
            raise ValueError(f"Invalid or too-short speech clip: {row['audio_path']}")
        encoded = self.extractor(waveform, sampling_rate=SAMPLE_RATE)
        return {
            "input_values": encoded.input_values[0],
            "labels": self.label_to_id[row["label"]],
        }


class AudioCollator:
    def __init__(self, extractor):
        self.extractor = extractor

    def __call__(self, examples):
        batch = self.extractor.pad(
            [{"input_values": item["input_values"]} for item in examples],
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([item["labels"] for item in examples], dtype=torch.long)
        return batch


class WeightedAudioTrainer(Trainer):
    """Use mild inverse-frequency weights for the rare calm/surprised classes."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        weights = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = CrossEntropyLoss(weight=weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def metrics_for(eval_prediction, id_to_eva):
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    actual = eval_prediction.label_ids
    predicted_output_ids = np.argmax(logits, axis=-1)
    predicted = np.asarray([
        LABEL_TO_ID[id_to_eva[int(output_id)]]
        for output_id in predicted_output_ids
    ])
    all_ids = list(range(len(LABELS)))
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=all_ids, average="macro", zero_division=0)),
    }


def keep_logits_only(logits, labels):
    """Drop optional hidden states/attentions before Trainer caches predictions."""
    if isinstance(logits, tuple):
        return logits[0]
    return logits


def evaluate_predictions(trainer, dataset, rows, id_to_eva, split: str) -> dict:
    prediction = trainer.predict(dataset, metric_key_prefix=split)
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    actual_ids = prediction.label_ids.tolist()
    predicted_output_ids = np.argmax(logits, axis=-1).tolist()
    predicted_ids = [LABEL_TO_ID[id_to_eva[int(index)]] for index in predicted_output_ids]
    labels = list(LABELS)
    perceived = [row["perceived_emotion"] for row in rows if row.get("perceived_emotion") in LABELS]
    self_reported = [row["label"] for row in rows if row.get("perceived_emotion") in LABELS]
    return {
        "split": split,
        "examples": len(rows),
        "accuracy": round(float(accuracy_score(actual_ids, predicted_ids)), 4),
        "macro_f1": round(float(f1_score(actual_ids, predicted_ids, labels=list(range(len(labels))), average="macro", zero_division=0)), 4),
        "labels": labels,
        "classification_report": classification_report(
            actual_ids, predicted_ids, labels=list(range(len(labels))), target_names=labels,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(actual_ids, predicted_ids, labels=list(range(len(labels)))).tolist(),
        "speaker_groups": len({row["speaker_group_id"] for row in rows}),
        "self_report_vs_perceived": {
            "count": len(perceived),
            "agreement": round(
                sum(left == right for left, right in zip(self_reported, perceived)) / len(perceived), 4
            ) if perceived else None,
            "note": "Perceived labels are optional and are not used as model training targets.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/eva-voice-affect"))
    parser.add_argument("--base-model", default="superb/wav2vec2-base-superb-er")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate --base-model on the held-out test split without training.")
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--unfreeze-feature-encoder", action="store_true", help="Fine-tune the convolutional audio feature encoder too.")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    splits = load_manifest(manifest)
    extractor = AutoFeatureExtractor.from_pretrained(args.base_model)
    if args.eval_only:
        model = AutoModelForAudioClassification.from_pretrained(args.base_model)
        id_to_eva = model_output_labels(model)
    else:
        id_to_eva = dict(enumerate(LABELS))
        model_config = AutoConfig.from_pretrained(args.base_model)
        # Transformers 5 removed the embedding state from the sequence of
        # hidden states used by Wav2Vec2 layer pooling. EVA's older checkpoint
        # has 13 pooling weights, which causes a 12-vs-13 runtime error there.
        # Keep the legacy pooling behavior on Transformers 4; with 5, use the
        # final encoder layer and save that choice in the new checkpoint.
        if int(TRANSFORMERS_VERSION.split(".", maxsplit=1)[0]) >= 5:
            model_config.use_weighted_layer_sum = False
        model_config.num_labels = len(LABELS)
        model_config.label2id = LABEL_TO_ID
        model_config.id2label = {index: label for index, label in id_to_eva.items()}
        model = AutoModelForAudioClassification.from_pretrained(
            args.base_model,
            config=model_config,
            ignore_mismatched_sizes=True,
        )
    label_to_id = LABEL_TO_ID
    model.config.id2label = {int(key): value for key, value in id_to_eva.items()}
    model.config.label2id = {label: index for index, label in id_to_eva.items()}
    if not args.unfreeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    datasets = {
        split: VoiceDataset(rows, label_to_id, extractor)
        for split, rows in splits.items()
    }
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=not args.eval_only,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        gradient_checkpointing=not args.eval_only,
        fp16=torch.cuda.is_available(),
        # Do not retain all output tensors on the GPU during long evaluation.
        eval_accumulation_steps=8,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
        data_seed=42,
    )
    train_counts = np.bincount(
        [label_to_id[row["label"]] for row in splits["train"]],
        minlength=len(LABELS),
    )
    class_weights = None
    if not args.eval_only:
        class_weights = torch.tensor(
            np.sqrt(train_counts.sum() / (len(LABELS) * train_counts)),
            dtype=torch.float32,
        )
        class_weights = class_weights / class_weights.mean()

    trainer = WeightedAudioTrainer(
        model=model,
        args=training_args,
        train_dataset=None if args.eval_only else datasets["train"],
        eval_dataset=datasets["dev"],
        data_collator=AudioCollator(extractor),
        compute_metrics=lambda prediction: metrics_for(prediction, id_to_eva),
        preprocess_logits_for_metrics=keep_logits_only,
        class_weights=class_weights,
    )
    if not args.eval_only:
        trainer.train()
        trainer.save_model(str(output_dir))
        extractor.save_pretrained(str(output_dir))

    test_metrics = evaluate_predictions(
        trainer, datasets["test"], splits["test"], id_to_eva, "test"
    )
    training_sources = {}
    for rows in splits.values():
        for row in rows:
            source = row["source_dataset"]
            training_sources[source] = {
                "source_license": row["source_license"],
            }
    report = {
        "base_model": args.base_model,
        "model_used_for_test": str(output_dir) if not args.eval_only else args.base_model,
        "label_order_by_model_id": {str(index): label for index, label in id_to_eva.items()},
        "canonical_labels": list(LABELS),
        "training_sources": training_sources,
        "frozen_feature_encoder": not args.unfreeze_feature_encoder,
        "train_examples": len(splits["train"]),
        "train_class_counts": {
            label: int(train_counts[label_to_id[label]])
            for label in LABELS
        },
        "class_weighting": "normalized square-root inverse frequency" if not args.eval_only else "none (baseline)",
        "dev_examples": len(splits["dev"]),
        "test": test_metrics,
        "warning": "This is a small-domain pilot metric. RAVDESS and CREMA-D are acted corpora; do not treat these results as natural-conversation accuracy or use test results to tune hyperparameters or thresholds.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / ("baseline-report.json" if args.eval_only else "test-report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
