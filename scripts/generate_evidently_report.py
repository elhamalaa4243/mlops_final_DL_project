"""Generate an Evidently 0.7.x data drift HTML report."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

FEATURE_NAMES = [
    "brightness",
    "contrast",
    "sharpness",
]


def load_dataframe(path):
    with np.load(path) as data:
        pixels = data["pixel_features"]
        confidence = data["confidence"]

    frame = pd.DataFrame(pixels, columns=FEATURE_NAMES)
    frame["confidence"] = confidence
    return frame


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
        default=Path("reports/evidently_drift.html"),
    )
    args = parser.parse_args()

    reference = load_dataframe(args.reference)
    current = load_dataframe(args.current)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference,
        current_data=current,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(args.output))

    print(f"PASS: Evidently report saved to {args.output}")
    print(f"Reference samples: {len(reference)}")
    print(f"Current samples: {len(current)}")
    print(f"Monitored features: {list(reference.columns)}")


if __name__ == "__main__":
    main()
