"""Benchmark OpenVINO CPU inference using the validated ONNX model."""

import json
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from PIL import Image

from pet_breed.inference import PetBreedPredictor

ONNX_MODEL = Path("models/onnx/resnet18_v1.onnx")
CHECKPOINT = Path("models/frozen/resnet18_v1.pt")
CLASSES = Path("data/processed/classes.json")
IMAGE_DIR = Path("data/raw/oxford-iiit-pet/images")
REPORT = Path("reports/openvino_cpu_benchmark.json")

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

    core = ov.Core()
    model = core.read_model(str(ONNX_MODEL))

    compiled_model = core.compile_model(
        model,
        "CPU",
        {
            "INFERENCE_NUM_THREADS": CPU_THREADS,
            "NUM_STREAMS": 1,
            "PERFORMANCE_HINT": "LATENCY",
        },
    )

    input_port = compiled_model.input(0)
    output_port = compiled_model.output(0)

    def run_openvino():
        return compiled_model({input_port: input_array})[output_port]

    results = measure(run_openvino)

    previous = json.loads(
        Path("reports/onnx_cpu_benchmark.json").read_text()
    )

    pytorch_p95 = previous["pytorch"]["p95_ms"]
    onnx_p95 = previous["onnxruntime"]["p95_ms"]
    openvino_p95 = results["p95_ms"]

    report = {
        "model": "resnet18_v1",
        "device": "CPU",
        "cpu_threads": CPU_THREADS,
        "batch_size": 1,
        "warmup_iterations": WARMUP,
        "measured_iterations": ITERATIONS,
        "measurement_scope": "model inference only; preprocessing excluded",
        "pytorch_p95_ms": pytorch_p95,
        "onnxruntime_p95_ms": onnx_p95,
        "openvino": results,
        "openvino_vs_pytorch_p95_speedup": pytorch_p95 / openvino_p95,
        "openvino_vs_onnx_p95_speedup": onnx_p95 / openvino_p95,
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
