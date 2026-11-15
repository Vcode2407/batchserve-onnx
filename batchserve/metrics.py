"""Per-process Prometheus metrics with fixed label cardinality."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "batchserve_requests_total", "Completed HTTP predictions by outcome", ["outcome"],
            registry=self.registry,
        )
        self.queued = Gauge(
            "batchserve_queue_depth", "Admitted predictions pending dispatch",
            registry=self.registry,
        )
        self.batch_size = Histogram(
            "batchserve_batch_size", "Predictions dispatched per runtime invocation",
            buckets=(1, 2, 4, 8, 16, 32, 64), registry=self.registry,
        )
        self.queue_seconds = Histogram(
            "batchserve_queue_seconds", "Admission to dispatch duration for dispatched predictions",
            buckets=(0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.256, 1),
            registry=self.registry,
        )
        self.runtime_seconds = Histogram(
            "batchserve_runtime_seconds", "Batch execution duration including thread scheduling",
            buckets=(0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064),
            registry=self.registry,
        )
        self.backend_errors = Counter(
            "batchserve_backend_errors_total", "Runtime invocations that failed",
            registry=self.registry,
        )
