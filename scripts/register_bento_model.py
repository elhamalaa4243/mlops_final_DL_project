"""Register the frozen ResNet-18 model in BentoML."""

import json
from pathlib import Path

import bentoml
import torch

from pet_breed.model import create_model

CHECKPOINT = Path("models/frozen/resnet18_v1.pt")
CLASSES = Path("data/processed/classes.json")


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not CLASSES.is_file():
        raise FileNotFoundError(f"Classes not found: {CLASSES}")

    class_names = json.loads(CLASSES.read_text(encoding="utf-8"))

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model = create_model(
        num_classes=len(class_names),
        pretrained=False,
        backbone=checkpoint["model_name"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    saved = bentoml.pytorch.save_model(
        "pet_breed_resnet18",
        model,
        signatures={
            "__call__": {
                "batchable": True,
                "batch_dim": 0,
            }
        },
        metadata={
            "model_version": CHECKPOINT.stem,
            "num_classes": len(class_names),
        },
    )

    print("Registered model:", saved.tag)


if __name__ == "__main__":
    main()
