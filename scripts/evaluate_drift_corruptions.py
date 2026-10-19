"""Evaluate calibrated drift detectors across synthetic corruptions."""

import csv
import json
from pathlib import Path

import numpy as np
from detect_drift_calibrated import calculate_mmd, load_features
from scipy.stats import wasserstein_distance

ROOT = Path("data/scoring/corruptions")
OUTPUT = Path("reports/detector_scorecard.csv")

SCENARIOS = [
    "clean_control",
    "brightness_low",
    "brightness_medium",
    "brightness_high",
    "contrast_low",
    "contrast_medium",
    "contrast_high",
    "blur_low",
    "blur_medium",
    "blur_high",
]

PIXEL_NAMES = ("brightness", "contrast", "sharpness")


def main():
    calibration = json.loads(
        Path("config/drift_thresholds.json").read_text()
    )

    reference = load_features(
        Path("data/scoring/reference_features.npz")
    )

    mean = np.asarray(calibration["mmd"]["normalization_mean"])
    std = np.asarray(calibration["mmd"]["normalization_std"])
    bandwidth = float(calibration["mmd"]["bandwidth"])
    pixel_std = np.asarray(calibration["pixel_reference_std"])
    thresholds = calibration["thresholds"]

    rng = np.random.default_rng(calibration["seed"])
    ref_idx = rng.choice(
        len(reference["embeddings"]),
        calibration["batch_size"],
        replace=False,
    )

    ref_embeddings = (
        reference["embeddings"][ref_idx] - mean
    ) / std

    results = []

    for scenario in SCENARIOS:
        current = load_features(ROOT / scenario / "features.npz")

        # Current batches contain exactly 500 samples.
        cur_idx = np.arange(len(current["embeddings"]))

        cur_embeddings = (
            current["embeddings"][cur_idx] - mean
        ) / std

        scores = {
            "embedding_mmd2": calculate_mmd(
                ref_embeddings, cur_embeddings, bandwidth
            ),
            "confidence_wasserstein": float(
                wasserstein_distance(
                    reference["confidence"][ref_idx],
                    current["confidence"][cur_idx],
                )
            ),
        }

        shifts = np.abs(
            reference["pixel_features"][ref_idx].mean(axis=0)
            - current["pixel_features"][cur_idx].mean(axis=0)
        ) / pixel_std

        for index, name in enumerate(PIXEL_NAMES):
            scores[f"pixel_{name}"] = float(shifts[index])

        alerts = {
            key: scores[key] > thresholds[key]
            for key in thresholds
        }

        row = {
            "scenario": scenario,
            "drift_detected": any(alerts.values()),
        }

        for key in thresholds:
            row[f"{key}_score"] = scores[key]
            row[f"{key}_alert"] = alerts[key]

        results.append(row)

    # Calculate change relative to the clean control.
    clean = results[0]

    for row in results:
        for key in thresholds:
            row[f"{key}_delta_vs_clean"] = (
                row[f"{key}_score"] - clean[f"{key}_score"]
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {OUTPUT}")
    print()

    for row in results:
        triggered = [
            key.replace("_alert", "")
            for key, value in row.items()
            if key.endswith("_alert") and value
        ]

        print(
            f"{row['scenario']:20s} "
            f"drift={row['drift_detected']!s:5s} "
            f"detectors={', '.join(triggered) or 'none'}"
        )


if __name__ == "__main__":
    main()
