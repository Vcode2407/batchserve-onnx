# CPU inference benchmark

The load generator compared two configurations of the same HTTP service and ONNX model. Each used one CPU inference session and one intra-op thread. The parameters that changed were maximum batch size and assembly wait.

| 32 concurrent clients, 1,024 requests per trial | Single-item | Dynamic batching |
| --- | ---: | ---: |
| Maximum batch size | 1 | 16 |
| Maximum assembly wait | 0 ms | 2 ms |
| Median successful predictions per second | 101.20 | 116.60 |
| Median trial p50 | 170.68 ms | 143.17 ms |
| Median trial p95 | 912.80 ms | 796.11 ms |
| Median runtime calls per trial | 1,024 | 450 |
| Median mean batch size | 1.00 | 2.28 |

All 1,024 calls in every timed trial returned HTTP 200. The harness compared every response to single-row ONNX execution for the same input, with relative tolerance 2e-4 and absolute tolerance 2e-5. The most direct effect was fewer runtime calls for the same amount of work. The latency and throughput readings include HTTP and vary across trials.

| Trial | Single predictions/s | Single p95 | Dynamic predictions/s | Dynamic p95 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 78.75 | 1,181.67 ms | 87.23 | 1,183.69 ms |
| 2 | 118.81 | 761.26 ms | 125.91 | 767.37 ms |
| 3 | 101.20 | 912.80 ms | 116.60 | 796.11 ms |

## Low load and overload

At one sequential client, both configurations used 64 invocations for 64 requests. Batching increased the observed median latency from **5.29 ms to 16.17 ms**. A lone request gains no coalescing benefit. The 2 ms setting controls the dispatch timer, and actual wake-up time depends on the event loop and OS scheduler.

A separate 384-request burst with 96 concurrent clients and a queue capacity of four returned **381 HTTP 200s and 3 HTTP 503s**. All 16 sequential probes after the burst succeeded, and the queue depth returned to zero. The test suite uses a controllably blocked model to distinguish overloaded requests from admitted requests that exceed their deadline.

## Method and reproduction

Measured on November 16, 2026, on Windows with an Intel Core i7-1185G7, Python 3.12.10, and ONNX Runtime 1.32.0. The service and HTTP load generator ran on the same host over loopback. Each mode warmed 128 requests before the parallel trials. The model is a seeded synthetic MLP with two 768-unit hidden layers, and the harness cycles through 32 generated inputs. Timers exclude correctness assertions and metrics scraping. Each client sends its next request after receiving the previous one, so this is a closed-loop test.

```sh
python -m pip install -r requirements-dev.txt
python -m scripts.benchmark --requests 1024 --concurrency 32 --trials 3
```
