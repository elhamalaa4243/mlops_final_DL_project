"""Fail CI when candidate top-1 accuracy is below the frozen baseline."""

import argparse
import json
import math
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Required file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")

    return data


def validate_accuracy(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")

    value = float(value)

    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between 0 and 1")

    return value


def check_quality(candidate_path: Path, baseline_path: Path) -> bool:
    candidate = load_json(candidate_path)
    baseline = load_json(baseline_path)

    if "test_accuracy" not in candidate:
        raise ValueError("Candidate metrics missing test_accuracy")

    if "minimum_top1_accuracy" not in baseline:
        raise ValueError("Baseline missing minimum_top1_accuracy")

    accuracy = validate_accuracy(
        candidate["test_accuracy"], "Candidate test_accuracy"
    )
    minimum = validate_accuracy(
        baseline["minimum_top1_accuracy"], "Baseline minimum_top1_accuracy"
    )

    print(f"Candidate Top-1: {accuracy:.6f} ({accuracy * 100:.2f}%)")
    print(f"Baseline Top-1:  {minimum:.6f} ({minimum * 100:.2f}%)")

    if accuracy < minimum:
        print("FAIL: Candidate Top-1 accuracy is below baseline.")
        return False

    print("PASS: Candidate Top-1 accuracy meets baseline.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("reports/dvc_metrics.json"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("config/model_quality_baseline.json"),
    )
    args = parser.parse_args()

    try:
        return 0 if check_quality(args.candidate, args.baseline) else 1
    except (TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
