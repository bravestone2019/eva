"""Download English and/or Hindi FLEURS into ignored local data directories."""

import argparse
from pathlib import Path

LANGUAGES = {"english": "en_us", "hindi": "hi_in"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["english", "hindi", "both"], default="both")
    parser.add_argument("--output-root", type=Path, default=Path("data/fleurs"))
    args = parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("Install optional training dependencies first: pip install -r requirements-training.txt") from error
    selected = LANGUAGES if args.language == "both" else {args.language: LANGUAGES[args.language]}
    for name, config in selected.items():
        target = args.output_root / name
        print(f"Downloading google/fleurs config={config} into {target}")
        dataset = load_dataset("google/fleurs", config)
        dataset.save_to_disk(str(target))
        print(dataset)


if __name__ == "__main__":
    main()
