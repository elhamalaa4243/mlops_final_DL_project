"""Evaluate synthetic corruption effects against matched clean images."""

import csv
import json
from pathlib import Path

import numpy as np
from detect_drift_calibrated import calculate_mmd, load_features
from scipy.stats import wasserstein_distance

ROOT = Path("data/scoring/corruptions")
OUTPUT = Path("reports/paired_corruption_scorecard.csv")

SCENARIOS = [
    "brightness_low", "brightness_medium", "brightness_high",
    "contrast_low", "contrast_medium", "contrast_high",
    "blur_low", "blur_medium", "blur_high",
]

PIXEL_NAMES = ("brightness", "contrast", "sharpness")


def main():
    calibration = json.loads(
        Path("config/drift_thresholds.json").read_text()
    )

    clean = load_features(ROOT / "clean_control" / "features.npz")

    mean = np.asarray(calibration["mmd"]["normalization_mean"])
    std = np.asarray(calibration["mmd"]["normalization_std"])
    bandwidth = float(calibration["mmd"]["bandwidth"])
    pixel_std = np.asarray(calibration["pixel_reference_std"])
    thresholds = calibration["thresholds"]

    clean_embeddings = (clean["embeddings"] - mean) / std

    results = []

    for scenario in SCENARIOS:
        corrupted = load_features(ROOT / scenario / "features.npz")

        if len(clean["embeddings"]) != len(corrupted["embeddings"]):
            raise ValueError(f"Sample count mismatch: {scenario}")

        corrupted_embeddings = (
            corrupted["embeddings"] - mean
        ) / std

        scores = {
            "embedding_mmd2": calculate_mmd(
                clean_embeddings, corrupted_embeddings, bandwidth
            ),
            "confidence_wasserstein": float(
                wasserstein_distance(
                    clean["confidence"],
                    corrupted["confidence"],
                )
            ),
        }

        shifts = np.abs(
            clean["pixel_features"].mean(axis=0)
            - corrupted["pixel_features"].mean(axis=0)
        ) / pixel_std

        for index, name in enumerate(PIXEL_NAMES):
            scores[f"pixel_{name}"] = float(shifts[index])

        alerts = {
            key: bool(scores[key] > thresholds[key])
            for key in thresholds
        }

        row = {
            "scenario": scenario,
            "any_alert": any(alerts.values()),
        }

        for key in thresholds:
            row[f"{key}_score"] = scores[key]
            row[f"{key}_alert"] = alerts[key]

        results.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {OUTPUT}\n")

    for row in results:
        detected = [
            key.removesuffix("_alert")
            for key, value in row.items()
            if key.endswith("_alert") and value
        ]

        print(
            f"{row['scenario']:20s} "
            f"alert={row['any_alert']!s:5s} "
            f"detectors={', '.join(detected) or 'none'}"
        )


if __name__ == "__main__":
    main()
