import csv
from pathlib import Path

def read_metrics(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return next(row for row in rows if row["Name"] == "Aggregated")

fastapi = read_metrics("reports/locust_fastapi_stats.csv")
bento = read_metrics("reports/locust_bentoml_stats.csv")

metrics = [
    ("Requests", "Request Count"),
    ("Failures", "Failure Count"),
    ("Average latency (ms)", "Average Response Time"),
    ("p50 latency (ms)", "50%"),
    ("p95 latency (ms)", "95%"),
    ("p99 latency (ms)", "99%"),
    ("Throughput (req/s)", "Requests/s"),
]

lines = [
    "# CPU Serving Benchmark",
    "",
    "Model: ResNet18 v1",
    "",
    "Load: 5 concurrent users, 1-2s wait time, 2 minutes per service.",
    "",
    "| Metric | FastAPI | BentoML |",
    "|---|---:|---:|",
]

for label, key in metrics:
    lines.append(f"| {label} | {fastapi[key]} | {bento[key]} |")

lines += [
    "",
    "## Interpretation",
    "",
    "FastAPI showed lower latency under the tested load.",
    "Both services completed the benchmark without HTTP failures.",
    "Throughput is load-limited and does not establish maximum capacity.",
    "BentoML dynamic batching has not yet been independently verified.",
    "The APIs use different request encodings: multipart vs raw JPEG.",
]

output = Path("reports/serving_comparison.md")
output.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(output.read_text())
