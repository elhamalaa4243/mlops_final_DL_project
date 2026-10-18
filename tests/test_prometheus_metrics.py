"""Tests for Prometheus HTTP metrics."""

import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from httpx import ASGITransport, AsyncClient
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from pet_breed.api import prometheus_middleware
from pet_breed.metrics import HTTP_DURATION, HTTP_REQUESTS, PREDICTIONS


def create_test_app():
    test_app = FastAPI()
    test_app.middleware("http")(prometheus_middleware)

    @test_app.get("/health")
    def health():
        return {"status": "healthy"}

    @test_app.post("/predict")
    def predict():
        PREDICTIONS.inc()
        return {"breed": "Havanese"}

    @test_app.get("/fail")
    def fail():
        raise HTTPException(status_code=422, detail="Invalid request")

    @test_app.get("/metrics")
    def metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return test_app


def metric_value(metric, labels=None):
    labels = labels or {}
    for sample in metric.collect()[0].samples:
        if (
            sample.name in {
                metric._name + "_total",
                metric._name + "_count",
            }
            and all(sample.labels.get(k) == v for k, v in labels.items())
        ):
            return sample.value
    return 0


def test_prometheus_instrumentation():
    async def run():
        app = create_test_app()
        transport = ASGITransport(app=app)

        labels_health = {
            "method": "GET",
            "endpoint": "/health",
            "status": "200",
        }
        labels_predict = {
            "method": "POST",
            "endpoint": "/predict",
            "status": "200",
        }
        labels_fail = {
            "method": "GET",
            "endpoint": "other",
            "status": "422",
        }

        before_health = metric_value(HTTP_REQUESTS, labels_health)
        before_predict = metric_value(HTTP_REQUESTS, labels_predict)
        before_fail = metric_value(HTTP_REQUESTS, labels_fail)
        before_predictions = metric_value(PREDICTIONS)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            assert (await client.get("/health")).status_code == 200
            assert (await client.post("/predict")).status_code == 200
            assert (await client.get("/fail")).status_code == 422

            response = await client.get("/metrics")
            assert response.status_code == 200
            assert "petbreed_http_requests_total" in response.text
            assert "petbreed_http_request_duration_seconds_bucket" in response.text

        assert metric_value(HTTP_REQUESTS, labels_health) == before_health + 1
        assert metric_value(HTTP_REQUESTS, labels_predict) == before_predict + 1
        assert metric_value(HTTP_REQUESTS, labels_fail) == before_fail + 1
        assert metric_value(PREDICTIONS) == before_predictions + 1

        # The /metrics endpoint must not instrument itself.
        assert not any(
            sample.labels.get("endpoint") == "/metrics"
            for sample in HTTP_REQUESTS.collect()[0].samples
        )

        assert any(
            sample.name == "petbreed_http_request_duration_seconds_count"
            and sample.labels.get("endpoint") == "/predict"
            for sample in HTTP_DURATION.collect()[0].samples
        )

    asyncio.run(run())
