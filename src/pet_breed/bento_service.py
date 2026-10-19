"""BentoML service using the registered batchable PyTorch Runner."""

import json
from pathlib import Path

import bentoml
import torch
from bentoml.io import JSON
from bentoml.io import Image as BentoImage
from PIL import Image

from pet_breed.transforms import get_eval_transforms

MODEL_TAG = "pet_breed_resnet18:hgmfvpwkxox6arzi"

model_ref = bentoml.pytorch.get(MODEL_TAG)
runner = model_ref.to_runner()

svc = bentoml.Service(
    "pet_breed_classifier",
    runners=[runner],
)

CLASS_NAMES = json.loads(
    Path("data/processed/classes.json").read_text(encoding="utf-8")
)

CALIBRATION = json.loads(
    Path("models/frozen/calibration.json").read_text(encoding="utf-8")
)

if CALIBRATION["model_version"] != model_ref.info.metadata["model_version"]:
    raise ValueError("Calibration model version does not match BentoML model")

TEMPERATURE = float(CALIBRATION["temperature"])
THRESHOLD = float(CALIBRATION["threshold"])
TRANSFORM = get_eval_transforms()

CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay",
    "British_Shorthair", "Egyptian_Mau", "Maine_Coon",
    "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx",
}


@svc.api(input=BentoImage(), output=JSON(), route="/predict")
async def predict(image: Image.Image):
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)

    # The registered Runner combines compatible requests along batch_dim=0.
    logits = await runner.async_run(tensor)
    logits = torch.as_tensor(logits).reshape(-1)

    probabilities = torch.softmax(logits / TEMPERATURE, dim=0)

    confidence, index = probabilities.max(dim=0)
    breed = CLASS_NAMES[index.item()]

    values, indices = torch.topk(probabilities, k=3)

    return {
        "breed": breed,
        "species": "cat" if breed in CAT_BREEDS else "dog",
        "confidence": float(confidence),
        "top_3": [
            {"breed": CLASS_NAMES[i], "confidence": float(p)}
            for p, i in zip(values.tolist(), indices.tolist())
        ],
        "decision": "accept" if float(confidence) >= THRESHOLD else "abstain",
        "model_version": CALIBRATION["model_version"],
    }
