"""Fine-tune EVA's multi-label text affect classifier on GoEmotions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_metrics(label_count):
    def compute_metrics(prediction):
        logits, labels = prediction
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        predicted = probabilities >= 0.5
        expected = labels >= 0.5
        true_positive = np.logical_and(predicted, expected).sum(axis=0)
        false_positive = np.logical_and(predicted, np.logical_not(expected)).sum(axis=0)
        false_negative = np.logical_and(np.logical_not(predicted), expected).sum(axis=0)
        micro_tp = true_positive.sum()
        micro_fp = false_positive.sum()
        micro_fn = false_negative.sum()
        micro_f1 = 2 * micro_tp / max(2 * micro_tp + micro_fp + micro_fn, 1)
        per_label_f1 = 2 * true_positive / np.maximum(
            2 * true_positive + false_positive + false_negative, 1
        )
        return {
            "micro_f1": float(micro_f1),
            "macro_f1": float(per_label_f1[:label_count].mean()),
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--validation-limit", type=int, default=3000)
    parser.add_argument("--test-limit", type=int, default=3000)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Load the downloaded GoEmotions DatasetDict from this local folder.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--eval-split",
        choices=("validation", "test"),
        default="test",
        help="Split to score when --eval-only is used (default: test).",
    )
    args = parser.parse_args()

    try:
        from datasets import Sequence, Value, load_dataset, load_from_disk
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise SystemExit("Install optional training dependencies first: pip install -r requirements-training.txt") from error

    if args.dataset_dir:
        dataset = load_from_disk(str(args.dataset_dir.resolve()))
    else:
        dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    label_names = dataset["train"].features["labels"].feature.names
    if "neutral" not in label_names:
        raise ValueError("GoEmotions dataset schema did not include the neutral label.")
    for split_name, limit in (
        ("train", args.train_limit),
        ("validation", args.validation_limit),
        ("test", args.test_limit),
    ):
        if split_name not in dataset:
            raise ValueError(f"GoEmotions dataset is missing {split_name!r} split.")
        if limit > 0 and len(dataset[split_name]) > limit:
            dataset[split_name] = dataset[split_name].shuffle(seed=42).select(range(limit))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label_names),
        id2label={index: label for index, label in enumerate(label_names)},
        label2id={label: index for index, label in enumerate(label_names)},
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    def encode(batch):
        encoded = tokenizer(batch["text"], truncation=True, max_length=args.max_length)
        multi_hot = np.zeros((len(batch["labels"]), len(label_names)), dtype=np.float32)
        for row, labels in enumerate(batch["labels"]):
            multi_hot[row, labels] = 1.0
        encoded["labels"] = multi_hot.tolist()
        return encoded

    encoded_dataset = dataset.map(
        encode,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing GoEmotions",
    )
    # The source schema stores labels as integer class IDs. Convert the
    # multi-hot vectors to floats for BCEWithLogitsLoss.
    encoded_dataset = encoded_dataset.cast_column(
        "labels", Sequence(Value("float32"))
    )
    output_dir = args.output_dir.resolve()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=not args.eval_only,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=42,
        data_seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None if args.eval_only else encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        compute_metrics=make_metrics(len(label_names)),
        processing_class=tokenizer,
    )
    if not args.eval_only:
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

    evaluation_split = args.eval_split if args.eval_only else "test"
    metrics = trainer.evaluate(
        encoded_dataset[evaluation_split],
        metric_key_prefix=f"eval_{evaluation_split}" if args.eval_only else "test",
    )
    metric_prefix = f"eval_{evaluation_split}" if args.eval_only else "test"
    report = {
        "dataset": "google-research-datasets/go_emotions (simplified)",
        "base_model": args.model,
        "checkpoint": args.model if args.eval_only else str(output_dir),
        "label_count": len(label_names),
        "labels": label_names,
        "multi_label": True,
        "threshold": 0.5,
        "split_rows": {name: len(encoded_dataset[name]) for name in ("train", "validation", "test")},
        "evaluation_split": evaluation_split,
        f"{metric_prefix}_micro_f1": metrics.get(f"{metric_prefix}_micro_f1"),
        f"{metric_prefix}_macro_f1": metrics.get(f"{metric_prefix}_macro_f1"),
        "warning": "GoEmotions is English Reddit text; test metrics do not establish performance on EVA users or other languages.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / ("baseline-report.json" if args.eval_only else "test-report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
