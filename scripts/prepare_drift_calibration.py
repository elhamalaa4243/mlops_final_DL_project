"""Create disjoint clean calibration batches from unused training images."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path", "label", "breed", "source_split", "split"
        ])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--size", type=int, default=500)
    args = parser.parse_args()

    train = [
        row for row in read_csv(Path("data/processed/manifest.csv"))
        if row["split"] == "train"
    ]

    reference = read_csv(Path("data/scoring/reference_manifest.csv"))
    reference_paths = {row["image_path"] for row in reference}

    unused = [
        row for row in train
        if row["image_path"] not in reference_paths
    ]

    if len(unused) < 2 * args.size:
        raise ValueError("Not enough unused training images")

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(unused))

    batch_a = [unused[int(i)] for i in indices[:args.size]]
    batch_b = [
        unused[int(i)]
        for i in indices[args.size:2 * args.size]
    ]

    output_dir = Path("data/scoring/calibration")
    write_csv(output_dir / "clean_a.csv", batch_a)
    write_csv(output_dir / "clean_b.csv", batch_b)

    all_paths = [
    {row["image_path"] for row in group}
    for group in (reference, batch_a, batch_b)
]

    assert not (all_paths[0] & all_paths[1])
    assert not (all_paths[0] & all_paths[2])
    assert not (all_paths[1] & all_paths[2])

    summary = {
        "seed": args.seed,
        "reference_count": len(reference),
        "clean_a_count": len(batch_a),
        "clean_b_count": len(batch_b),
        "overlap": 0,
        "unused_train_images_remaining": len(unused) - 2 * args.size,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
