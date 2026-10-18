"""Compute independent embedding, pixel, and confidence drift scores."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance


PIXEL_NAMES = ("brightness", "contrast", "sharpness")


def rbf_mmd_squared(reference, current, max_samples=500, seed=42):
    """Unbiased RBF-MMD squared using a median-distance bandwidth."""
    rng = np.random.default_rng(seed)

    if len(reference) > max_samples:
        reference = reference[
            rng.choice(len(reference), max_samples, replace=False)
        ]
    if len(current) > max_samples:
        current = current[
            rng.choice(len(current), max_samples, replace=False)
        ]

    # Normalize each dimension using reference statistics only.
    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    std = np.maximum(std, 1e-6)

    x = (reference - mean) / std
    y = (current - mean) / std

    combined = np.vstack([x, y])
    squared_norms = np.sum(combined**2, axis=1)

    distances = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2 * combined @ combined.T
    )
    distances = np.maximum(distances, 0.0)

    upper = distances[np.triu_indices(len(combined), k=1)]
    positive = upper[upper > 0]
    bandwidth = float(np.median(positive))

    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError("Invalid RBF bandwidth")

    kernel = np.exp(-distances / bandwidth)

    n = len(x)
    m = len(y)

    k_xx = kernel[:n, :n]
    k_yy = kernel[n:, n:]
    k_xy = kernel[:n, n:]

    xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    xy = k_xy.mean()

    return float(xx + yy - 2 * xy), bandwidth


def pixel_drift(reference, current):
    results = {}

    for index, name in enumerate(PIXEL_NAMES):
        ref_values = reference[:, index]
        cur_values = current[:, index]

        ref_mean = float(ref_values.mean())
        cur_mean = float(cur_values.mean())
        ref_std = max(float(ref_values.std()), 1e-6)

        results[name] = {
            "reference_mean": ref_mean,
            "current_mean": cur_mean,
            "standardized_mean_shift": float(
                abs(cur_mean - ref_mean) / ref_std
            ),
        }

    return results


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
        "--output",
        type=Path,
        default=Path("reports/drift_scores.json"),
    )
    args = parser.parse_args()

    with np.load(args.reference) as ref_data:
        reference = {
            key: ref_data[key].copy()
            for key in ("embeddings", "pixel_features", "confidence")
        }

    with np.load(args.current) as cur_data:
        current = {
            key: cur_data[key].copy()
            for key in ("embeddings", "pixel_features", "confidence")
        }

    for name, features in [
        ("reference", reference),
        ("current", current),
    ]:
        for key, values in features.items():
            if not np.isfinite(values).all():
                raise ValueError(f"{name}.{key} contains invalid values")

    mmd, bandwidth = rbf_mmd_squared(
        reference["embeddings"].astype(np.float64),
        current["embeddings"].astype(np.float64),
    )

    pixel = pixel_drift(
        reference["pixel_features"],
        current["pixel_features"],
    )

    confidence_distance = float(
        wasserstein_distance(
            reference["confidence"],
            current["confidence"],
        )
    )

    results = {
        "reference_samples": len(reference["confidence"]),
        "current_samples": len(current["confidence"]),
        "embedding_drift": {
            "method": "RBF-MMD squared (unbiased)",
            "score": mmd,
            "rbf_bandwidth": bandwidth,
            "samples_per_group": min(
                len(reference["embeddings"]), 500
            ),
        },
        "pixel_drift": pixel,
        "confidence_drift": {
            "method": "Wasserstein distance",
            "score": confidence_distance,
            "reference_mean_confidence": float(
                reference["confidence"].mean()
            ),
            "current_mean_confidence": float(
                current["confidence"].mean()
            ),
        },
        "alert_status": "not_calibrated",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
