"""Validate ONNX Runtime numerical parity against frozen PyTorch model."""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from pet_breed.inference import PetBreedPredictor


CHECKPOINT = Path("models/frozen/resnet18_v1.pt")
CLASSES = Path("data/processed/classes.json")
ONNX_MODEL = Path("models/onnx/resnet18_v1.onnx")
IMAGE_DIR = Path("data/raw/oxford-iiit-pet/images")
REPORT = Path("reports/onnx_parity.json")

NUM_IMAGES = 200
TOLERANCE = 1e-4


def main():
    onnx.checker.check_model(str(ONNX_MODEL))

    predictor = PetBreedPredictor(
        checkpoint_path=CHECKPOINT,
        classes_path=CLASSES,
        device="cpu",
    )

    session = ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    images = sorted(IMAGE_DIR.glob("*.jpg"))[:NUM_IMAGES]

    if len(images) != NUM_IMAGES:
        raise RuntimeError(
            f"Expected {NUM_IMAGES} images, found {len(images)}"
        )

    max_diff = 0.0
    top1_matches = 0

    for index, image_path in enumerate(images, start=1):
        with Image.open(image_path) as image:
            tensor = predictor.transform(
                image.convert("RGB")
            ).unsqueeze(0)

        with torch.inference_mode():
            pytorch_logits = predictor.model(tensor).cpu().numpy()

        onnx_logits = session.run(
            None,
            {input_name: tensor.numpy()},
        )[0]

        if pytorch_logits.shape != onnx_logits.shape:
            raise AssertionError(
                f"Output shape mismatch: "
                f"{pytorch_logits.shape} vs {onnx_logits.shape}"
            )

        diff = float(
            np.max(np.abs(pytorch_logits - onnx_logits))
        )

        max_diff = max(max_diff, diff)

        if np.argmax(pytorch_logits) == np.argmax(onnx_logits):
            top1_matches += 1

        if index % 50 == 0:
            print(
                f"Validated {index}/{NUM_IMAGES} images | "
                f"max_diff={max_diff:.8f}"
            )

    passed = max_diff <= TOLERANCE

    result = {
        "pytorch_model": str(CHECKPOINT),
        "onnx_model": str(ONNX_MODEL),
        "images_tested": NUM_IMAGES,
        "max_absolute_logit_difference": max_diff,
        "tolerance": TOLERANCE,
        "top1_agreement": top1_matches / NUM_IMAGES,
        "status": "PASS" if passed else "FAIL",
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== ONNX PARITY RESULT ===")
    print(json.dumps(result, indent=2))

    if not passed:
        raise SystemExit("ONNX numerical parity FAILED")


if __name__ == "__main__":
    main()
