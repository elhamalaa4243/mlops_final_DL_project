
"""Unit tests for the model quality gate."""

import json

import pytest

from scripts.check_model_quality import check_quality


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def baseline(tmp_path):
    return write_json(
        tmp_path / "baseline.json",
        {"minimum_top1_accuracy": 0.8743526846552194},
    )


def test_quality_gate_passes_above_baseline(tmp_path, baseline):
    candidate = write_json(
        tmp_path / "candidate.json",
        {"test_accuracy": 0.90},
    )
    assert check_quality(candidate, baseline)


def test_quality_gate_passes_equal_baseline(tmp_path, baseline):
    candidate = write_json(
        tmp_path / "candidate.json",
        {"test_accuracy": 0.8743526846552194},
    )
    assert check_quality(candidate, baseline)


def test_quality_gate_fails_below_baseline(tmp_path, baseline):
    candidate = write_json(
        tmp_path / "candidate.json",
        {"test_accuracy": 0.85},
    )
    assert not check_quality(candidate, baseline)


def test_quality_gate_rejects_missing_accuracy(tmp_path, baseline):
    candidate = write_json(tmp_path / "candidate.json", {})
    with pytest.raises(ValueError, match="missing test_accuracy"):
        check_quality(candidate, baseline)


@pytest.mark.parametrize(
    ("accuracy", "exception"),
    [
        (-0.1, ValueError),
        (1.1, ValueError),
        (float("nan"), ValueError),
        (True, TypeError),
        ("0.90", TypeError),
    ],
)
def test_quality_gate_rejects_invalid_accuracy(
    tmp_path, baseline, accuracy, exception
):
    candidate = write_json(
        tmp_path / "candidate.json",
        {"test_accuracy": accuracy},
    )

    with pytest.raises(exception):
        check_quality(candidate, baseline)
