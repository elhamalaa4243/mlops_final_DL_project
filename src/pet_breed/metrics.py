"""Prometheus metrics for the Pet Breed API."""

from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "petbreed_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

HTTP_DURATION = Histogram(
    "petbreed_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(
        0.01, 0.025, 0.05, 0.075, 0.1,
        0.15, 0.2, 0.3, 0.5, 1.0, 2.5, 5.0,
    ),
)

PREDICTIONS = Counter(
    "petbreed_predictions_total",
    "Successful predictions",
)
