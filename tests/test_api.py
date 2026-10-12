"""API contract and reference-logit tests against the frozen checkpoint."""
import io
from pathlib import Path

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from pet_breed.api import app
from pet_breed.transforms import get_eval_transforms


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_version": "resnet18_v1", "num_classes": 37}


def test_predict_valid_image(client):
    image_path = pd.read_csv("data/processed/manifest.csv").iloc[0]["image_path"]
    response = client.post("/predict", files={"file": ("pet.jpg", Path(image_path).read_bytes(), "image/jpeg")})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["top_3"]) == 3
    assert payload["species"] in {"cat", "dog"}
    assert payload["decision"] in {"accept", "abstain"}


def test_reject_text(client):
    response = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert response.status_code == 422


def test_reject_fake_image(client):
    response = client.post("/predict", files={"file": ("fake.jpg", b"not a jpeg", "image/jpeg")})
    assert response.status_code == 422


def test_reject_oversized_image(client):
    response = client.post("/predict", files={"file": ("huge.jpg", b"0" * (10 * 1024 * 1024 + 1), "image/jpeg")})
    assert response.status_code == 422


def test_reject_rgba_png(client):
    buf = io.BytesIO()
    Image.new("RGBA", (24, 24), (1, 2, 3, 4)).save(buf, format="PNG")
    response = client.post("/predict", files={"file": ("rgba.png", buf.getvalue(), "image/png")})
    assert response.status_code == 422


def test_reference_logits_agree_with_api_path(client):
    predictor = app.state.predictor
    image_path = pd.read_csv("data/processed/manifest.csv").iloc[0]["image_path"]
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    tensor = get_eval_transforms()(image).unsqueeze(0)
    with torch.inference_mode():
        reference_logits = predictor.model(tensor).squeeze(0).cpu()
    serving_logits = predictor.logits(image)
    torch.testing.assert_close(reference_logits, serving_logits, atol=1e-4, rtol=0)
