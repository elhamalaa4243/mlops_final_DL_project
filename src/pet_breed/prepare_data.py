
from pathlib import Path
from collections import Counter
import json
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.datasets import OxfordIIITPet


SEED = 42
DATA_ROOT = Path("data/raw")
OUTPUT_DIR = Path("data/processed")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(split: str):
    return OxfordIIITPet(
        root=str(DATA_ROOT),
        split=split,
        target_types="category",
        download=True,
    )


def build_manifest(dataset, split_name: str):
    records = []

    for index in range(len(dataset)):
        image_path = Path(dataset._images[index])
        label = int(dataset._labels[index])

        records.append({
            "image_path": str(image_path.resolve()),
            "label": label,
            "breed": dataset.classes[label],
            "source_split": split_name,
        })

    return pd.DataFrame(records)


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading/loading Oxford-IIIT Pet dataset...")

    trainval = load_dataset("trainval")
    test = load_dataset("test")

    print(f"Trainval images: {len(trainval)}")
    print(f"Test images: {len(test)}")
    print(f"Number of breeds: {len(trainval.classes)}")

    trainval_df = build_manifest(trainval, "trainval")
    test_df = build_manifest(test, "test")

    train_df, val_df = train_test_split(
        trainval_df,
        test_size=0.2,
        random_state=SEED,
        stratify=trainval_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    manifest = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True,
    )

    manifest.to_csv(OUTPUT_DIR / "manifest.csv", index=False)

    with open(OUTPUT_DIR / "classes.json", "w") as f:
        json.dump(trainval.classes, f, indent=2)

    print("\nDataset split summary:")
    print(manifest["split"].value_counts())

    print("\nTraining breed distribution:")
    print(Counter(train_df["breed"]))

    print("\nSaved:")
    print(OUTPUT_DIR / "manifest.csv")
    print(OUTPUT_DIR / "classes.json")


if __name__ == "__main__":
    main()
