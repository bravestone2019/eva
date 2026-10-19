"""Download GoEmotions, HinEmo, and OpenAssistant OASST1 for local experiments."""

import argparse
from pathlib import Path


DATASETS = {
    "go_emotions": ("google-research-datasets/go_emotions", "simplified"),
    "hinemo": ("sosang11/hinemo", None),
    "oasst1": ("OpenAssistant/oasst1", None),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets to download (default: all three).",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            "Install optional training dependencies first: "
            "pip install -r requirements-training.txt"
        ) from error

    for name in args.datasets:
        repository, config = DATASETS[name]
        target = args.output_root / name
        if (target / "dataset_dict.json").exists() or (target / "dataset_info.json").exists():
            print(f"Already downloaded: {target}; skipping")
            continue

        print(f"Downloading {repository}" + (f" config={config}" if config else ""))
        dataset = load_dataset(repository, config) if config else load_dataset(repository)
        dataset.save_to_disk(str(target))
        print(f"Saved {name} to {target}")
        print(dataset)


if __name__ == "__main__":
    main()
