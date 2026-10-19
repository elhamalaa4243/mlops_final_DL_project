"""Automated numerical parity validation for the exported ONNX model."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_MODEL = PROJECT_ROOT / "models/onnx/resnet18_v1.onnx"
CHECKPOINT = PROJECT_ROOT / "models/frozen/resnet18_v1.pt"
IMAGE_DIR = PROJECT_ROOT / "data/raw/oxford-iiit-pet/images"
PARITY_SCRIPT = PROJECT_ROOT / "scripts/validate_onnx.py"
PARITY_REPORT = PROJECT_ROOT / "reports/onnx_parity.json"


@pytest.mark.slow
def test_onnx_numerical_parity():
    """Assert PyTorch/ONNX logits differ by at most 1e-4 on 200 images."""

    if not ONNX_MODEL.is_file():
        pytest.skip("ONNX model is not available")

    if not CHECKPOINT.is_file():
        pytest.skip("Frozen checkpoint is not available")

    if not IMAGE_DIR.is_dir():
        pytest.skip("Oxford-IIIT Pet images are not available")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        [sys.executable, str(PARITY_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    assert result.returncode == 0, (
        f"ONNX parity script failed:\n"
        f"{result.stdout}\n{result.stderr}"
    )

    report = json.loads(PARITY_REPORT.read_text(encoding="utf-8"))

    assert report["images_tested"] == 200
    assert report["max_absolute_logit_difference"] <= 1e-4
    assert report["top1_agreement"] == 1.0
    assert report["status"] == "PASS"
