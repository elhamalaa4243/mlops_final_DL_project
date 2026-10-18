"""Compare PyTorch and ONNX Runtime CPU inference latency."""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from pet_breed.inference import PetBreedPredictor


CHECKPOINT = Path("models/frozen/resnet18_v1.pt")
CLASSES = Path("data/processed/classes.json")
ONNX_MODEL = Path("models/onnx/resnet18_v1.onnx")
IMAGE_DIR = Path("data/raw/oxford-iiit-pet/images")
REPORT = Path("reports/onnx_cpu_benchmark.json")

WARMUP = 20
ITERATIONS = 200
CPU_THREADS = 1


def measure(fn):
    for _ in range(WARMUP):
        fn()

    times = []

    for _ in range(ITERATIONS):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
    }


def main():
    torch.set_num_threads(CPU_THREADS)

    predictor = PetBreedPredictor(
        checkpoint_path=CHECKPOINT,
        classes_path=CLASSES,
        device="cpu",
    )

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(IMAGE_DIR)

    with Image.open(image_paths[0]) as image:
        tensor = predictor.transform(image.convert("RGB")).unsqueeze(0)

    input_array = tensor.numpy()

    options = ort.SessionOptions()
    options.intra_op_num_threads = CPU_THREADS
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(ONNX_MODEL),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    def run_pytorch():
        with torch.inference_mode():
            return predictor.model(tensor)

    def run_onnx():
        return session.run(None, {input_name: input_array})

    pytorch_results = measure(run_pytorch)
    onnx_results = measure(run_onnx)

    report = {
        "model": "resnet18_v1",
        "device": "CPU",
        "cpu_threads": CPU_THREADS,
        "batch_size": 1,
        "warmup_iterations": WARMUP,
        "measured_iterations": ITERATIONS,
        "measurement_scope": "model inference only; preprocessing excluded",
        "pytorch": pytorch_results,
        "onnxruntime": onnx_results,
        "p95_speedup": (
            pytorch_results["p95_ms"] / onnx_results["p95_ms"]
        ),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
