"""Expose the latest completed batch drift report as Prometheus metrics."""

import json
import math
import os
from datetime import datetime
from pathlib import Path

from prometheus_client import Gauge

DETECTORS = (
    "embedding_mmd2",
    "confidence_wasserstein",
    "pixel_brightness",
    "pixel_contrast",
    "pixel_sharpness",
)

DRIFT_SCORE = Gauge(
    "petbreed_drift_score",
    "Latest completed batch drift score",
    ["detector"],
)

DRIFT_THRESHOLD = Gauge(
    "petbreed_drift_threshold",
    "Calibrated drift detection threshold",
    ["detector"],
)

DRIFT_ALERT = Gauge(
    "petbreed_drift_alert",
    "Drift alert per detector: 1 means alert",
    ["detector"],
)

DRIFT_DETECTED = Gauge(
    "petbreed_drift_detected",
    "Overall drift detected in latest completed batch",
)

DRIFT_BATCH_TIMESTAMP = Gauge(
    "petbreed_drift_batch_timestamp_seconds",
    "Completion time of latest monitored batch as Unix timestamp",
)

DRIFT_REPORT_AVAILABLE = Gauge(
    "petbreed_drift_report_available",
    "1 if a valid completed drift report is available",
)


def _latest_completed_run(root: Path):
    candidates = []

    for summary_path in root.glob("*/monitoring_summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            completed_at = datetime.fromisoformat(
                summary["completed_at"]
            )
            if completed_at.tzinfo is None:
                continue

            report_path = summary_path.parent / "drift_scores.json"
            if report_path.is_file():
                candidates.append((completed_at, report_path))
        except (OSError, ValueError, KeyError, TypeError):
            continue

    return max(candidates, default=None, key=lambda item: item[0])


def refresh_drift_metrics():
    """Refresh gauges from the latest valid completed monitoring batch."""
    root = Path(os.getenv("DRIFT_RUNS_DIR", "data/scoring/runs"))
    latest = _latest_completed_run(root)

    if latest is None:
        DRIFT_REPORT_AVAILABLE.set(0)
        return

    completed_at, report_path = latest

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))

        scores = report["scores"]
        thresholds = report["thresholds"]
        alerts = report["alerts"]

        values = []
        for detector in DETECTORS:
            score = float(scores[detector])
            threshold = float(thresholds[detector])
            alert = alerts[detector]

            if not math.isfinite(score) or not math.isfinite(threshold):
                raise ValueError("Non-finite drift metric")
            if not isinstance(alert, bool):
                raise TypeError("Invalid drift alert")

            values.append((detector, score, threshold, int(alert)))

        overall = report["drift_detected"]
        if not isinstance(overall, bool):
            raise TypeError("Invalid overall drift status")

    except (OSError, ValueError, KeyError, TypeError):
        DRIFT_REPORT_AVAILABLE.set(0)
        return

    for detector, score, threshold, alert in values:
        DRIFT_SCORE.labels(detector=detector).set(score)
        DRIFT_THRESHOLD.labels(detector=detector).set(threshold)
        DRIFT_ALERT.labels(detector=detector).set(alert)

    DRIFT_DETECTED.set(int(overall))
    DRIFT_BATCH_TIMESTAMP.set(completed_at.timestamp())
    DRIFT_REPORT_AVAILABLE.set(1)
