"""Unit tests for dataset manifest generation without external data."""

from pathlib import Path
from types import SimpleNamespace

from pet_breed.prepare_data import build_manifest


def test_build_manifest_creates_correct_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    images_dir = tmp_path / "data" / "raw" / "images"
    images_dir.mkdir(parents=True)

    image_paths = [
        images_dir / "cat_1.jpg",
        images_dir / "dog_1.jpg",
        images_dir / "cat_2.jpg",
    ]

    for image_path in image_paths:
        image_path.write_bytes(b"fake-image")

    dataset = SimpleNamespace(
        _images=image_paths,
        _labels=[0, 1, 0],
        classes=["Abyssinian", "beagle"],
    )

    # build_manifest uses len(dataset), so supply a small dataset object.
    class FakeDataset:
        def __init__(self, data):
            self._images = data._images
            self._labels = data._labels
            self.classes = data.classes

        def __len__(self):
            return len(self._images)

    manifest = build_manifest(FakeDataset(dataset), "trainval")

    assert len(manifest) == 3
    assert manifest["image_path"].tolist() == [
        "data/raw/images/cat_1.jpg",
        "data/raw/images/dog_1.jpg",
        "data/raw/images/cat_2.jpg",
    ]
    assert manifest["label"].tolist() == [0, 1, 0]
    assert manifest["breed"].tolist() == [
        "Abyssinian",
        "beagle",
        "Abyssinian",
    ]
    assert manifest["source_split"].tolist() == ["trainval"] * 3
    assert manifest["image_path"].is_unique

    for image_path in manifest["image_path"]:
        assert Path(image_path).is_file()
