"""Run batch scoring and drift monitoring as one reproducible workflow."""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def run_command(command):
    print("\nRUN:", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/scoring/current_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scoring/monitored_batch"),
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")

    def execute(script, *arguments):
        command = [
            sys.executable,
            str(Path("scripts") / script),
            *map(str, arguments),
        ]
        print("\nRUN:", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=env)

    # 1. Score the incoming batch.
    execute(
        "batch_score.py",
        "--manifest", args.manifest,
        "--output-dir", output,
    )

    predictions = output / "predictions.csv"
    if not predictions.is_file():
        raise FileNotFoundError(
            f"Batch scoring did not produce {predictions}"
        )

    # 2. Extract features from exactly the same input manifest.
    current_features = output / "features.npz"

    execute(
        "extract_drift_features.py",
        "--manifest", args.manifest,
        "--output", current_features,
    )

    # 3. Run calibrated drift detection.
    drift_report = output / "drift_scores.json"

    execute(
        "detect_drift_calibrated.py",
        "--reference", "data/scoring/reference_features.npz",
        "--current", current_features,
        "--output", drift_report,
    )

    # 4. Generate Evidently HTML.
    evidently_report = output / "evidently_drift.html"

    execute(
        "generate_evidently_report.py",
        "--reference", "data/scoring/reference_features.npz",
        "--current", current_features,
        "--output", evidently_report,
    )

    # 5. Record workflow metadata.
    drift = json.loads(drift_report.read_text(encoding="utf-8"))

    summary = {
        "completed_at": datetime.now(UTC).isoformat(),
        "manifest": str(args.manifest),
        "predictions": str(predictions),
        "features": str(current_features),
        "drift_report": str(drift_report),
        "evidently_report": str(evidently_report),
        "drift_detected": drift["drift_detected"],
        "alerts": drift["alerts"],
    }

    summary_path = output / "monitoring_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPASS: Batch monitoring completed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
