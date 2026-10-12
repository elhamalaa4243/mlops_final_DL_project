FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CHECKPOINT=/app/models/frozen/resnet18_v1.pt \
    MODEL_CLASSES=/app/data/processed/classes.json \
    MODEL_CALIBRATION=/app/models/frozen/calibration.json

WORKDIR /app

COPY requirements-serving.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-serving.txt

COPY pyproject.toml ./
COPY src/ ./src/
RUN python -m pip install --no-deps --no-build-isolation .

COPY data/processed/classes.json ./data/processed/classes.json
COPY models/frozen/resnet18_v1.pt ./models/frozen/resnet18_v1.pt
COPY models/frozen/calibration.json ./models/frozen/calibration.json

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1
CMD ["uvicorn", "pet_breed.api:app", "--host", "0.0.0.0", "--port", "8000"]
