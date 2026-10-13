"""Register and verify a calibrated pet-breed classifier in MLflow 3.x.

 .`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from PIL import Image


class PetBreedPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from pet_breed.inference import PetBreedPredictor

        self.predictor = PetBreedPredictor(
            context.artifacts["checkpoint"], context.artifacts["classes"], device="cpu"
        )
        calibration = json.loads(Path(context.artifacts["calibration"]).read_text())
        self.temperature = float(calibration["temperature"])
        self.threshold = float(calibration["threshold"])

    def predict(self, context, model_input, params=None):
        if "image_path" not in model_input.columns:
            raise ValueError("Expected a DataFrame containing an image_path column")
        predictions = []
        for image_path in model_input["image_path"]:
            with Image.open(str(image_path)) as image:
                predictions.append(
                    self.predictor.predict(
                        image, temperature=self.temperature, threshold=self.threshold
                    )
                )
        return pd.DataFrame(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--registered-name", default="PetBreedClassifier")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--sample-image", help="Optional local image for round-trip verification")
    args = parser.parse_args()

    for path in (args.checkpoint, args.classes, args.calibration):
        if not Path(path).is_file():
            parser.error(f"Missing file: {path}")

    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != "resnet50" or checkpoint["num_classes"] != 37:
        parser.error("Expected a 37-class ResNet-50 checkpoint")

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    run = client.get_run(args.run_id)
    if run.info.status != "FINISHED":
        parser.error("Source training run must be FINISHED")

    # MLflow 3.x logs the model as its own artifact with a model ID.
    with mlflow.start_run(run_id=args.run_id):
        info = mlflow.pyfunc.log_model(
            name="pet_breed_pyfunc",
            python_model=PetBreedPyFunc(),
            artifacts={
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "classes": str(Path(args.classes).resolve()),
                "calibration": str(Path(args.calibration).resolve()),
            },
            pip_requirements=["mlflow==3.17.0", "torch", "torchvision", "pillow", "pandas"],
            code_paths=[str(Path("src").resolve())],
            registered_model_name=args.registered_name,
            await_registration_for=300,
        )

    print("Logged model URI:", info.model_uri)
    versions = client.search_model_versions(f"name='{args.registered_name}'")
    matches = [v for v in versions if v.source == info.model_uri or v.run_id == args.run_id]
    if not matches:
        print("Registration submitted; check MLflow Models for the new version")
        return
    version = max(matches, key=lambda v: int(v.version))
    print(f"Registered model: {args.registered_name}, version {version.version}")

    if args.sample_image:
        from pet_breed.inference import PetBreedPredictor

        predictor = PetBreedPredictor(args.checkpoint, args.classes)
        calibration = json.loads(Path(args.calibration).read_text())
        with Image.open(args.sample_image) as image:
            expected = predictor.predict(
                image,
                temperature=float(calibration["temperature"]),
                threshold=float(calibration["threshold"]),
            )
        loaded = mlflow.pyfunc.load_model(f"models:/{args.registered_name}/{version.version}")
        actual = loaded.predict(pd.DataFrame({"image_path": [args.sample_image]})).iloc[0].to_dict()
        assert actual["breed"] == expected["breed"]
        assert actual["decision"] == expected["decision"]
        assert abs(float(actual["confidence"]) - expected["confidence"]) < 1e-5
        print("Registry round-trip prediction: PASS")
    else:
        print("Supply --sample-image to verify Registry loading and prediction")
    print("Production alias NOT assigned; verify before promotion")


if __name__ == "__main__":
    main()
