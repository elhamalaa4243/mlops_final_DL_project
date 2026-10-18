"""Extract ResNet18 embeddings, pixel features, and calibrated confidence."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pet_breed.model import create_model
from pet_breed.transforms import get_eval_transforms


def pixel_features(image):
    gray = np.asarray(
        image.convert("L").resize((224, 224)),
        dtype=np.float32,
    )

    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4 * center
    )

    return [
        float(gray.mean()),
        float(gray.std()),
        float(laplacian.var()),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    classes = json.loads(
        Path("data/processed/classes.json").read_text(encoding="utf-8")
    )
    calibration = json.loads(
        Path("models/frozen/calibration.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        "models/frozen/resnet18_v1.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = create_model(
        num_classes=len(classes),
        pretrained=False,
        backbone=checkpoint["model_name"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ResNet18's fc receives the 512-dimensional penultimate representation.
    embeddings = []

    def capture_embedding(module, inputs):
        embeddings.append(inputs[0].detach().cpu().numpy().reshape(-1))

    hook = model.fc.register_forward_pre_hook(capture_embedding)

    transform = get_eval_transforms()
    temperature = float(calibration["temperature"])

    image_names = []
    pixels = []
    confidence = []

    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            image_path = Path(row["image_path"])

            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                pixels.append(pixel_features(rgb))
                tensor = transform(rgb).unsqueeze(0)

            logits = model(tensor)
            probabilities = torch.softmax(logits / temperature, dim=1)

            image_names.append(str(image_path))
            confidence.append(float(probabilities.max().item()))

            if index % 100 == 0:
                print(f"Extracted {index}/{len(rows)}", flush=True)

    hook.remove()

    features = np.stack(embeddings).astype(np.float32)

    if features.shape != (len(rows), 512):
        raise ValueError(f"Unexpected embedding shape: {features.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output,
        image_paths=np.asarray(image_names),
        embeddings=features,
        pixel_features=np.asarray(pixels, dtype=np.float32),
        confidence=np.asarray(confidence, dtype=np.float32),
    )

    print(f"Saved: {args.output}")
    print(f"Embeddings shape: {features.shape}")


if __name__ == "__main__":
    main()
