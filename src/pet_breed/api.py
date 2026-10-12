"""FastAPI service with a single startup-loaded model and validated uploads."""
from __future__ import annotations

import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from pet_breed.inference import PetBreedPredictor
from pet_breed.schemas import HealthResponse, PredictionResponse

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = PetBreedPredictor(
        os.getenv("MODEL_CHECKPOINT", "models/frozen/resnet18_v1.pt"),
        os.getenv("MODEL_CLASSES", "data/processed/classes.json"),
    )
    calibration_path = Path(os.getenv("MODEL_CALIBRATION", "models/frozen/calibration.json"))
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        app.state.temperature = float(calibration["temperature"])
        app.state.threshold = float(calibration["threshold"])
        if not __import__("math").isfinite(app.state.temperature) or not __import__("math").isfinite(app.state.threshold) or app.state.temperature <= 0 or not 0 <= app.state.threshold <= 1:
            raise ValueError("Invalid calibration configuration")
        if calibration.get("model_version") != app.state.predictor.model_version:
            raise ValueError("Calibration file is for a different model version")
    #else:
        # Explicitly uncalibrated provisional defaults; fit on validation before release.
        #app.state.temperature = 1.0
        #app.state.threshold = 0.80
    else:
         raise RuntimeError(
        f"Required calibration file not found: {calibration_path}"
                            )
    yield


app = FastAPI(title="Pet Breed Classification", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    predictor = app.state.predictor
    return {"status": "healthy", "model_version": predictor.model_version,
            "num_classes": len(predictor.class_names)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=422, detail="Only JPEG and PNG images are supported")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=422, detail="Image exceeds 10 MiB limit")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.mode not in {"RGB", "L"}:
                raise HTTPException(status_code=422, detail="Only RGB or grayscale images are supported")
            image = opened.convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=422, detail="Invalid or unsafe image") from None
    return app.state.predictor.predict(
        image, temperature=app.state.temperature, threshold=app.state.threshold
    )
