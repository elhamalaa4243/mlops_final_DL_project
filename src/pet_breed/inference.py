"""Single shared model-loading and inference path for the API and tests."""
###Load the frozen model once, run shared preprocessing and inference
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from pet_breed.model import create_model
from pet_breed.transforms import get_eval_transforms


class PetBreedPredictor:
    def __init__(self, checkpoint_path: str | Path, classes_path: str | Path, *, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.class_names: list[str] = json.loads(Path(classes_path).read_text(encoding="utf-8"))
        if len(self.class_names) != 37 or len(set(self.class_names)) != 37:
            raise ValueError("Expected 37 unique class names in dataset index order")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        if checkpoint["num_classes"] != len(self.class_names):
            raise ValueError("Checkpoint output classes do not match label map")
        self.model = create_model(
                           num_classes=len(self.class_names),
                           pretrained=False,
                           backbone=checkpoint["model_name"],
                                 )
        # self.model = create_model(num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        self.transform = get_eval_transforms()
        self.model_version = self.checkpoint_path.stem

    def logits(self, image: Image.Image) -> torch.Tensor:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return self.model(tensor).squeeze(0).cpu()

    def predict(self, image: Image.Image, *, temperature: float = 1.0, threshold: float = 0.80) -> dict:
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        probabilities = torch.softmax(self.logits(image) / temperature, dim=0)
        confidence, index = probabilities.max(dim=0)
        breed = self.class_names[index.item()]
        # Oxford-IIIT Pet's 12 cat breeds are listed explicitly to avoid case-based heuristics.
        cat_breeds = {
            "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", "Egyptian_Mau",
            "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx",
        }
        species = "cat" if breed in cat_breeds else "dog"
        values, indices = torch.topk(probabilities, k=3)
        return {
            "breed": breed,
            "species": species,
            "confidence": float(confidence),
            "top_3": [
                {"breed": self.class_names[i], "confidence": float(p)}
                for p, i in zip(values.tolist(), indices.tolist())
            ],
            "decision": "accept" if float(confidence) >= threshold else "abstain",
            "model_version": self.model_version,
        }
