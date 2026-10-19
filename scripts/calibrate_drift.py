"""Calibrate drift thresholds using repeated clean-data comparisons."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

SEED = 42
N_REPEATS = 200
BATCH_SIZE = 500

FEATURE_NAMES = ("brightness", "contrast", "sharpness")


def load_features(path):
    with np.load(path) as data:
        return {
            "embeddings": data["embeddings"].astype(np.float64),
            "pixel_features": data["pixel_features"].astype(np.float64),
            "confidence": data["confidence"].astype(np.float64),
        }


def main():
    reference = load_features("data/scoring/reference_features.npz")
    clean_a = load_features(
        "data/scoring/calibration/clean_a_features.npz"
    )
    clean_b = load_features(
        "data/scoring/calibration/clean_b_features.npz"
    )

    # Fit normalization exclusively on the reference dataset.
    mean = reference["embeddings"].mean(axis=0)
    std = np.maximum(
        reference["embeddings"].std(axis=0),
        1e-6,
    )

    ref_embeddings = (reference["embeddings"] - mean) / std

    # Fixed RBF bandwidth, estimated from reference samples only.
    rng = np.random.default_rng(SEED)
    bandwidth_indices = rng.choice(
        len(ref_embeddings), 500, replace=False
    )
    bandwidth_samples = ref_embeddings[bandwidth_indices]
    distances = cdist(
        bandwidth_samples,
        bandwidth_samples,
        metric="sqeuclidean",
    )
    positive = distances[np.triu_indices(500, k=1)]
    bandwidth = float(np.median(positive[positive > 0]))

    # Pool clean images for repeated disjoint comparisons.
    pool = {
        key: np.concatenate(
            [reference[key], clean_a[key], clean_b[key]],
            axis=0,
        )
        for key in reference
    }

    normalized_embeddings = (pool["embeddings"] - mean) / std

    # Precompute kernel once for fast repeated MMD calculations.
    kernel = np.exp(
        -cdist(
            normalized_embeddings,
            normalized_embeddings,
            metric="sqeuclidean",
        ) / bandwidth
    )

    pixel_std = np.maximum(
        reference["pixel_features"].std(axis=0),
        1e-6,
    )

    scores = {
        "embedding_mmd2": [],
        "confidence_wasserstein": [],
        **{f"pixel_{name}": [] for name in FEATURE_NAMES},
    }

    n = BATCH_SIZE

    for _ in range(N_REPEATS):
        indices = rng.choice(len(pool["confidence"]), 2 * n, replace=False)
        x_idx = indices[:n]
        y_idx = indices[n:]

        k_xx = kernel[np.ix_(x_idx, x_idx)]
        k_yy = kernel[np.ix_(y_idx, y_idx)]
        k_xy = kernel[np.ix_(x_idx, y_idx)]

        xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        yy = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
        mmd = float(xx + yy - 2 * k_xy.mean())

        scores["embedding_mmd2"].append(mmd)

        scores["confidence_wasserstein"].append(
            float(
                wasserstein_distance(
                    pool["confidence"][x_idx],
                    pool["confidence"][y_idx],
                )
            )
        )

        x_pixels = pool["pixel_features"][x_idx]
        y_pixels = pool["pixel_features"][y_idx]

        shifts = np.abs(
            x_pixels.mean(axis=0) - y_pixels.mean(axis=0)
        ) / pixel_std

        for i, name in enumerate(FEATURE_NAMES):
            scores[f"pixel_{name}"].append(float(shifts[i]))

    thresholds = {
        key: float(np.quantile(values, 0.95))
        for key, values in scores.items()
    }

    output = {
        "method": "clean_resampling_95th_percentile",
        "seed": SEED,
        "n_repeats": N_REPEATS,
        "batch_size": BATCH_SIZE,
        "reference_samples": len(reference["confidence"]),
        "clean_pool_samples": len(pool["confidence"]),
        "mmd": {
            "kernel": "rbf",
            "bandwidth": bandwidth,
            "normalization_mean": mean.tolist(),
            "normalization_std": std.tolist(),
        },
        "pixel_reference_std": pixel_std.tolist(),
        "thresholds": thresholds,
        "limitations": (
            "Exploratory thresholds; repeated samples are not "
            "independent and do not establish production false-alarm rates."
        ),
    }

    output_path = Path("config/drift_thresholds.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("Calibration completed")
    print("Fixed RBF bandwidth:", bandwidth)
    print("\nThresholds:")
    for key, value in thresholds.items():
        print(f"  {key}: {value:.8f}")
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
