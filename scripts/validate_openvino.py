"""Validate OpenVINO numerical parity against frozen PyTorch model."""

import json
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from PIL import Image

from pet_breed.inference import PetBreedPredictor

CHECKPOINT = Path("models/frozen/resnet18_v1.pt")
CLASSES = Path("data/processed/classes.json")
ONNX_MODEL = Path("models/onnx/resnet18_v1.onnx")
IMAGE_DIR = Path("data/raw/oxford-iiit-pet/images")
REPORT = Path("reports/openvino_parity.json")

NUM_IMAGES = 200
TOLERANCE = 1e-4


def main():
    torch.set_num_threads(1)

    predictor = PetBreedPredictor(
        checkpoint_path=CHECKPOINT,
        classes_path=CLASSES,
        device="cpu",
    )

    core = ov.Core()
    model = core.read_model(str(ONNX_MODEL))
    compiled_model = core.compile_model(
        model,
        "CPU",
        {
            "INFERENCE_NUM_THREADS": 1,
            "NUM_STREAMS": 1,
            "PERFORMANCE_HINT": "LATENCY",
        },
    )

    input_port = compiled_model.input(0)
    output_port = compiled_model.output(0)

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

        openvino_logits = compiled_model(
            {input_port: tensor.numpy()}
        )[output_port]

        if pytorch_logits.shape != openvino_logits.shape:
            raise AssertionError(
                f"Output shape mismatch: "
                f"{pytorch_logits.shape} vs {openvino_logits.shape}"
            )

        diff = float(
            np.max(np.abs(pytorch_logits - openvino_logits))
        )

        max_diff = max(max_diff, diff)

        if np.argmax(pytorch_logits) == np.argmax(openvino_logits):
            top1_matches += 1

        if index % 50 == 0:
            print(
                f"Validated {index}/{NUM_IMAGES} images | "
                f"max_diff={max_diff:.8f}"
            )

    passed = max_diff <= TOLERANCE

    result = {
        "pytorch_model": str(CHECKPOINT),
        "openvino_source_model": str(ONNX_MODEL),
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

    print("\n=== OPENVINO PARITY RESULT ===")
    print(json.dumps(result, indent=2))

    if not passed:
        raise SystemExit("OpenVINO numerical parity FAILED")


if __name__ == "__main__":
    main()
