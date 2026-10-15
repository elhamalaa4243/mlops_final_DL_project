
from pathlib import Path

import pandas as pd


def test_dataset_splits():
    manifest_path = Path("data/processed/manifest.csv")
    assert manifest_path.exists(), "Run dataset preparation first"

    df = pd.read_csv(manifest_path)

    assert len(df) == 7349
    assert set(df["split"]) == {"train", "val", "test"}
    assert df["image_path"].is_unique
    assert df["label"].nunique() == 37

    for image_path in df["image_path"]:
        assert Path(image_path).is_file(), image_path
