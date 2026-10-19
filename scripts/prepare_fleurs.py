"""Create a smaller FLEURS train split while preserving official validation/test."""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as error:
        raise SystemExit("Install optional training dependencies first: pip install -r requirements-training.txt") from error

    dataset = load_from_disk(str(args.input_dir.resolve()))
    train = dataset["train"].shuffle(seed=args.seed)
    train = train.select(range(min(args.train_limit, len(train))))
    prepared = DatasetDict({
        "train": train,
        "validation": dataset["validation"],
        "test": dataset["test"],
    })
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    prepared.save_to_disk(str(args.output_dir.resolve()))
    print(prepared)


if __name__ == "__main__":
    main()
