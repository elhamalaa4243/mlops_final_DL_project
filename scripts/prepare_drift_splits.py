"""Prepare reproducible, disjoint drift reference and current batches."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def stratified_sample(rows, sample_size, rng):
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    total = len(rows)
    if sample_size > total:
        raise ValueError(
            f"Requested {sample_size} samples from {total} available"
        )

    # Allocate samples approximately proportional to class frequency.
    quotas = {
        label: sample_size * len(group) / total
        for label, group in by_label.items()
    }
    counts = {
        label: int(quota)
        for label, quota in quotas.items()
    }

    remaining = sample_size - sum(counts.values())
    labels = sorted(
        quotas,
        key=lambda label: (
            -(quotas[label] - counts[label]),
            label,
        ),
    )

    for label in labels[:remaining]:
        counts[label] += 1

    selected = []
    for label in sorted(by_label):
        group = by_label[label]
        indices = rng.choice(
            len(group),
            size=counts[label],
            replace=False,
        )
        selected.extend(group[int(i)] for i in indices)

    rng.shuffle(selected)
    return selected


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path",
        "label",
        "breed",
        "source_split",
        "split",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.csv"),
    )
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scoring"),
    )
    args = parser.parse_args()

    with args.manifest.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        rows = list(csv.DictReader(f))

    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]

    rng = np.random.default_rng(args.seed)

    reference = stratified_sample(train, args.size, rng)
    current = stratified_sample(test, args.size, rng)

    reference_paths = {r["image_path"] for r in reference}
    current_paths = {r["image_path"] for r in current}

    overlap = reference_paths & current_paths
    if overlap:
        raise RuntimeError(
            f"Reference/current overlap: {len(overlap)} images"
        )

    missing = [
        r["image_path"]
        for r in reference + current
        if not Path(r["image_path"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image files are missing; first: {missing[0]}"
        )

    write_manifest(
        args.output_dir / "reference_manifest.csv",
        reference,
    )
    write_manifest(
        args.output_dir / "current_manifest.csv",
        current,
    )

    summary = {
        "seed": args.seed,
        "reference": {
            "source_split": "train",
            "count": len(reference),
            "breeds": len(Counter(r["breed"] for r in reference)),
        },
        "current": {
            "source_split": "test",
            "count": len(current),
            "breeds": len(Counter(r["breed"] for r in current)),
        },
        "overlap": len(overlap),
    }

    output = args.output_dir / "drift_splits_summary.json"
    output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
