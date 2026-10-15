"""Real-model integration test for serving-logit parity."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from pet_breed.inference import PetBreedPredictor
from pet_breed.transforms import get_eval_transforms


@pytest.mark.integration
def test_reference_logits_agree_with_api_path():
    checkpoint = Path("models/frozen/resnet18_v1.pt")
    classes = Path("data/processed/classes.json")
    manifest = Path("data/processed/manifest.csv")

    if not all(path.exists() for path in (checkpoint, classes, manifest)):
        pytest.skip("Model or dataset artifacts unavailable")

    predictor = PetBreedPredictor(str(checkpoint), str(classes))
    image_path = pd.read_csv(manifest).iloc[0]["image_path"]

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")

    tensor = get_eval_transforms()(image).unsqueeze(0)

    with torch.inference_mode():
        reference_logits = predictor.model(tensor).squeeze(0).cpu()

    serving_logits = predictor.logits(image)

    torch.testing.assert_close(
        reference_logits,
        serving_logits,
        atol=1e-4,
        rtol=0,
    )