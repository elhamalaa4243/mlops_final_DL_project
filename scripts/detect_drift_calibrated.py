"""Detect drift using fixed reference-fitted calibration parameters."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


PIXEL_NAMES = ("brightness", "contrast", "sharpness")


def load_features(path):
    with np.load(path) as data:
        return {
            "embeddings": data["embeddings"].astype(np.float64),
            "pixel_features": data["pixel_features"].astype(np.float64),
            "confidence": data["confidence"].astype(np.float64),
        }


def calculate_mmd(x, y, bandwidth):
    """Unbiased RBF-MMD squared."""
    n, m = len(x), len(y)

    if n < 2 or m < 2:
        raise ValueError("MMD requires at least 2 samples per group")

    k_xx = np.exp(-cdist(x, x, "sqeuclidean") / bandwidth)
    k_yy = np.exp(-cdist(y, y, "sqeuclidean") / bandwidth)
    k_xy = np.exp(-cdist(x, y, "sqeuclidean") / bandwidth)

    xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))

    return float(xx + yy - 2 * k_xy.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/scoring/reference_features.npz"),
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("data/scoring/current_features.npz"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("config/drift_thresholds.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/drift_calibrated.json"),
    )
    args = parser.parse_args()

    reference = load_features(args.reference)
    current = load_features(args.current)
    calibration = json.loads(args.calibration.read_text())

    mean = np.asarray(
        calibration["mmd"]["normalization_mean"]
    )
    std = np.asarray(
        calibration["mmd"]["normalization_std"]
    )
    bandwidth = float(calibration["mmd"]["bandwidth"])
    pixel_std = np.asarray(calibration["pixel_reference_std"])
    thresholds = calibration["thresholds"]

    # Fixed, deterministic sampling.
    rng = np.random.default_rng(calibration["seed"])
    max_samples = int(calibration["batch_size"])

    ref_idx = rng.choice(
        len(reference["embeddings"]),
        min(len(reference["embeddings"]), max_samples),
        replace=False,
    )
    cur_idx = rng.choice(
        len(current["embeddings"]),
        min(len(current["embeddings"]), max_samples),
        replace=False,
    )

    x = (reference["embeddings"][ref_idx] - mean) / std
    y = (current["embeddings"][cur_idx] - mean) / std

    scores = {
        "embedding_mmd2": calculate_mmd(x, y, bandwidth),
        "confidence_wasserstein": float(
    wasserstein_distance(
        reference["confidence"][ref_idx],
        current["confidence"][cur_idx],
                        )
                                        ),
    }

    pixel_shift = np.abs(
    reference["pixel_features"][ref_idx].mean(axis=0)
    - current["pixel_features"][cur_idx].mean(axis=0)
) / pixel_std

    for index, name in enumerate(PIXEL_NAMES):
        scores[f"pixel_{name}"] = float(pixel_shift[index])

    alerts = {
        name: bool(score > thresholds[name])
        for name, score in scores.items()
    }

    result = {
        "reference_samples": len(reference["confidence"]),
        "current_samples": len(current["confidence"]),
        "scores": scores,
        "thresholds": thresholds,
        "alerts": alerts,
        "drift_detected": any(alerts.values()),
        "calibration_method": calibration["method"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
