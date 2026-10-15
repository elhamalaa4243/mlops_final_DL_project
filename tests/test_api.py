"""API contract tests without external model artifacts."""

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pet_breed.api import app


class FakePredictor:
    model_version = "resnet18_v1"

    def __init__(self, *args, **kwargs):
        self.class_names = [f"breed_{i}" for i in range(37)]

    def predict(self, image, temperature, threshold):
        return {
            "breed": "Abyssinian",
            "species": "cat",
            "confidence": 0.95,
            "top_3": [
                {"breed": "Abyssinian", "confidence": 0.95},
                {"breed": "Bengal", "confidence": 0.03},
                {"breed": "Birman", "confidence": 0.02},
            ],
            "decision": "accept",
            "model_version": self.model_version,
        }


@pytest.fixture
def client(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        '{"temperature": 1.0, "threshold": 0.8, "model_version": "resnet18_v1"}',
        encoding="utf-8",
    )

    with (
        patch("pet_breed.api.PetBreedPredictor", FakePredictor),
        patch.dict("os.environ", {"MODEL_CALIBRATION": str(calibration)}),
        TestClient(app) as test_client,
    ):
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_version": "resnet18_v1",
        "num_classes": 37,
    }


def test_predict_valid_image(client):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (100, 120, 140)).save(buf, format="JPEG")

    response = client.post(
        "/predict",
        files={"file": ("pet.jpg", buf.getvalue(), "image/jpeg")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["top_3"]) == 3
    assert payload["species"] in {"cat", "dog"}
    assert payload["decision"] in {"accept", "abstain"}


def test_reject_text(client):
    response = client.post(
        "/predict", files={"file": ("a.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 422


def test_reject_fake_image(client):
    response = client.post(
        "/predict", files={"file": ("fake.jpg", b"not a jpeg", "image/jpeg")}
    )
    assert response.status_code == 422


def test_reject_oversized_image(client):
    response = client.post(
        "/predict",
        files={"file": ("huge.jpg", b"0" * (10 * 1024 * 1024 + 1), "image/jpeg")},
    )
    assert response.status_code == 422


def test_reject_rgba_png(client):
    buf = io.BytesIO()
    Image.new("RGBA", (24, 24), (1, 2, 3, 4)).save(buf, format="PNG")

    response = client.post(
        "/predict",
        files={"file": ("rgba.png", buf.getvalue(), "image/png")},
    )
    assert response.status_code == 422