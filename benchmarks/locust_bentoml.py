"""Load test BentoML's raw-image prediction endpoint."""

import os
from pathlib import Path

from locust import HttpUser, between, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data/raw/oxford-iiit-pet/images"

configured = os.getenv("LOCUST_IMAGE")

if configured:
    IMAGE_PATH = Path(configured).expanduser().resolve()
else:
    images = sorted(IMAGE_DIR.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No JPG images found in {IMAGE_DIR}")
    IMAGE_PATH = images[0]

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

IMAGE_BYTES = IMAGE_PATH.read_bytes()

print(f"Benchmark image: {IMAGE_PATH}")


class PetBreedUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict(self):
        with self.client.post(
            "/predict",
            data=IMAGE_BYTES,
            headers={"Content-Type": "image/jpeg"},
            name="POST /predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
            else:
                response.success()
