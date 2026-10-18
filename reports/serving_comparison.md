# CPU Serving Benchmark

Model: ResNet18 v1

Load: 5 concurrent users, 1-2s wait time, 2 minutes per service.

| Metric | FastAPI | BentoML |
|---|---:|---:|
| Requests | 373 | 368 |
| Failures | 0 | 0 |
| Average latency (ms) | 74.85931036971935 | 97.8053490108402 |
| p50 latency (ms) | 65 | 83 |
| p95 latency (ms) | 130 | 160 |
| p99 latency (ms) | 280 | 400 |
| Throughput (req/s) | 3.133293158589212 | 3.117116470729411 |

## Interpretation

FastAPI showed lower latency under the tested load.
Both services completed the benchmark without HTTP failures.
Throughput is load-limited and does not establish maximum capacity.
BentoML dynamic batching has not yet been independently verified.
The APIs use different request encodings: multipart vs raw JPEG.
