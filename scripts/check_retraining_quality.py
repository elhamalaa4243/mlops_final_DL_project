"""Validate a newly evaluated candidate against the frozen production model."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_accuracy(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")

    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")

    return value


def check_candidate(metrics_path, checkpoint_path, baseline_path, tolerance):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    if not checkpoint_path.is_file():
        raise ValueError("Candidate checkpoint does not exist")

    actual_sha = sha256_file(checkpoint_path)
    if metrics["checkpoint_sha256"] != actual_sha:
        raise ValueError("Candidate checkpoint SHA256 mismatch")

    if Path(metrics["checkpoint"]).resolve() != checkpoint_path.resolve():
        raise ValueError("Evaluation report belongs to a different checkpoint")

    expected_samples = int(baseline["num_samples"])
    if metrics["test_samples"] != expected_samples:
        raise ValueError(
            f"Unexpected evaluation sample count: {metrics['test_samples']}"
        )

    if metrics["model_name"] != "resnet18":
        raise ValueError("Retraining candidate must use resnet18")

    candidate_accuracy = valid_accuracy(
        metrics["test_accuracy"], "Candidate accuracy"
    )
    production_accuracy = valid_accuracy(
        baseline["minimum_top1_accuracy"], "Production accuracy"
    )

    required_accuracy = production_accuracy - tolerance
    passed = candidate_accuracy >= required_accuracy

    print(f"Production accuracy: {production_accuracy:.6f}")
    print(f"Candidate accuracy:  {candidate_accuracy:.6f}")
    print(f"Required accuracy:   {required_accuracy:.6f}")
    print(f"Quality gate: {'PASS' if passed else 'FAIL'}")

    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("config/model_quality_baseline.json"),
    )
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()

    try:
        if not math.isfinite(args.tolerance) or not 0 <= args.tolerance <= 1:
            raise ValueError("Invalid tolerance")

        passed = check_candidate(
            args.metrics,
            args.checkpoint,
            args.baseline,
            args.tolerance,
        )
        return 0 if passed else 1

    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
