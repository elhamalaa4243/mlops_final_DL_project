"""Batch scoring for Oxford-IIIT Pet images."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image

from pet_breed.model import create_model
from pet_breed.transforms import get_eval_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/raw/oxford-iiit-pet/images"),
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CSV manifest containing image_path column",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scoring/output"),
    )
    args = parser.parse_args()

    if args.limit < 1000:
        raise ValueError("At least 1000 images are required")

    checkpoint_path = Path("models/frozen/resnet18_v1.pt")
    classes_path = Path("data/processed/classes.json")

    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        checkpoint_path,
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

    calibration = json.loads(
        Path("models/frozen/calibration.json").read_text(encoding="utf-8")
    )
    if calibration["model_version"] != checkpoint_path.stem:
        raise ValueError("Calibration model version mismatch")

    temperature = float(calibration["temperature"])
    threshold = float(calibration["threshold"])

    transform = get_eval_transforms()

    if args.manifest is not None:
        with args.manifest.open(
            newline="", encoding="utf-8-sig"
        ) as f:
            reader = csv.DictReader(f)
            if "image_path" not in (reader.fieldnames or []):
                raise ValueError("Manifest requires image_path column")
            images = [Path(row["image_path"]) for row in reader]
        images = images[:args.limit]
    else:
        images = sorted(args.image_dir.glob("*.jpg"))[:args.limit]

    if len(images) < args.limit:
        raise ValueError(
            f"Only {len(images)} images found; requested {args.limit}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output_dir / "predictions.csv"
    start = time.perf_counter()

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "predicted_breed",
                "confidence",
                "decision",
                "model_version",
            ],
        )
        writer.writeheader()

        with torch.inference_mode():
            for index, image_path in enumerate(images, start=1):
                with Image.open(image_path) as image:
                    tensor = transform(image.convert("RGB")).unsqueeze(0)

                logits = model(tensor)
                probabilities = torch.softmax(
                    logits / temperature, dim=1
                )
                confidence, predicted_index = probabilities.max(dim=1)

                writer.writerow(
                    {
                        "image": image_path.name,
                        "predicted_breed": classes[predicted_index.item()],
                        "confidence": float(confidence.item()),
                        "decision": (
                            "accept"
                            if float(confidence.item()) >= threshold
                            else "abstain"
                        ),
                        "model_version": checkpoint_path.stem,
                    }
                )

                if index % 100 == 0:
                    print(f"Scored {index}/{len(images)} images", flush=True)

    duration = time.perf_counter() - start

    summary = {
        "model_version": checkpoint_path.stem,
        "images_scored": len(images),
        "duration_seconds": round(duration, 2),
        "output_file": str(output_path),
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
