"""Train a separate Hindi/code-mixed text-emotion pilot on the local HinEmo data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def make_metrics(label_count: int):
    def compute_metrics(prediction):
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels = prediction
        predicted = np.argmax(logits, axis=-1)
        label_ids = list(range(label_count))
        return {
            "accuracy": float(accuracy_score(labels, predicted)),
            "macro_f1": float(
                f1_score(labels, predicted, labels=label_ids, average="macro", zero_division=0)
            ),
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/hinemo"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/eva-hinemo-pilot"))
    parser.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-limit", type=int, default=0, help="0 uses the full split.")
    parser.add_argument("--validation-limit", type=int, default=0, help="0 uses the full split.")
    parser.add_argument("--test-limit", type=int, default=0, help="0 uses the full split.")
    args = parser.parse_args()

    try:
        import torch
        from datasets import load_from_disk
        from sklearn.metrics import classification_report, confusion_matrix
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise SystemExit(
            "Install optional training dependencies first: "
            "pip install -r requirements-training.txt"
        ) from error

    dataset = load_from_disk(str(args.dataset_dir.resolve()))
    required_splits = ("train", "validation", "test")
    if any(split not in dataset for split in required_splits):
        raise ValueError("HinEmo must contain train, validation, and test splits.")
    required_columns = {"text", "gpt_emotion"}
    missing = required_columns.difference(dataset["train"].column_names)
    if missing:
        raise ValueError(f"HinEmo is missing columns: {', '.join(sorted(missing))}")

    label_names = sorted({str(label).strip().lower() for label in dataset["train"]["gpt_emotion"]})
    if len(label_names) != 4:
        raise ValueError(f"Expected the four documented HinEmo labels; found {label_names}.")
    label_to_id = {label: index for index, label in enumerate(label_names)}

    for split, limit in zip(required_splits, (args.train_limit, args.validation_limit, args.test_limit)):
        if limit > 0 and len(dataset[split]) > limit:
            dataset[split] = dataset[split].shuffle(seed=42).select(range(limit))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label_names),
        id2label={index: label for index, label in enumerate(label_names)},
        label2id=label_to_id,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )

    def encode(batch):
        texts = [str(text) for text in batch["text"]]
        encoded = tokenizer(texts, truncation=True, max_length=args.max_length)
        encoded["labels"] = [label_to_id[str(label).strip().lower()] for label in batch["gpt_emotion"]]
        return encoded

    encoded = dataset.map(
        encode,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing HinEmo",
    )
    output_dir = args.output_dir.resolve()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        eval_accumulation_steps=8,
        report_to="none",
        seed=42,
        data_seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        compute_metrics=make_metrics(len(label_names)),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    prediction = trainer.predict(encoded["test"], metric_key_prefix="test")
    predicted = np.argmax(prediction.predictions, axis=-1)
    labels = prediction.label_ids
    language_metrics = {}
    for language in sorted(set(str(value) for value in dataset["test"]["indiclid_label"])):
        indexes = [
            index
            for index, value in enumerate(dataset["test"]["indiclid_label"])
            if str(value) == language
        ]
        if not indexes:
            continue
        language_metrics[language] = {
            "examples": len(indexes),
            "accuracy": float((predicted[indexes] == labels[indexes]).mean()),
            "macro_f1": float(
                make_metrics(len(label_names))(
                    (prediction.predictions[indexes], labels[indexes])
                )["macro_f1"]
            ),
        }
    report = {
        "dataset": "sosang11/hinemo",
        "label_source": "gpt_emotion (machine-generated labels; treat as a noisy pilot target)",
        "base_model": args.model,
        "checkpoint": str(output_dir),
        "labels": label_names,
        "single_label": True,
        "language_note": (
            "HinEmo is described as Hindi/English code-mixed, but its IndicLID metadata is multilingual. "
            "Inspect language-tagged results before claiming Hindi-specific accuracy."
        ),
        "split_rows": {split: len(encoded[split]) for split in required_splits},
        "gpt_emotion_counts": {
            split: dict(Counter(str(label).strip().lower() for label in dataset[split]["gpt_emotion"]))
            for split in required_splits
        },
        "test_accuracy": float((predicted == labels).mean()),
        "test_macro_f1": float(
            make_metrics(len(label_names))((prediction.predictions, labels))["macro_f1"]
        ),
        "classification_report": classification_report(
            labels,
            predicted,
            labels=list(range(len(label_names))),
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=list(range(len(label_names)))).tolist(),
        "test_by_indiclid_language": language_metrics,
        "warning": "This pilot measures agreement with HinEmo's GPT-generated labels, not verified user emotion.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "test-report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
