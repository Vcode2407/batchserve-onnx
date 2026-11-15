"""Compare one-at-a-time and batched CPU inference over loopback HTTP."""

import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import tempfile
import time

import httpx
import numpy as np
from prometheus_client.parser import text_string_to_metric_families

from batchserve.model import FEATURES, OnnxModel

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"Content-Type": "application/octet-stream", "X-Timeout-Ms": "3000"}


@contextmanager
def server(batch, waiting_ms, queue):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = os.environ | {
        "BS_BATCH_SIZE": str(batch), "BS_MAX_WAIT_MS": str(waiting_ms),
        "BS_QUEUE_CAPACITY": str(queue), "BS_INTRA_OP_THREADS": "1",
        "BS_DEADLINE_GUARD_MS": "2",
    }
    with tempfile.TemporaryFile() as logs:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "batchserve.api:app", "--host", "127.0.0.1", "--port", str(port),
             "--no-access-log", "--log-level", "warning"],
            cwd=ROOT, env=env, stdout=logs, stderr=logs,
        )
        try:
            with httpx.Client(base_url=url, timeout=1, trust_env=False) as client:
                for _ in range(120):
                    if process.poll() is not None:
                        break
                    try:
                        if client.get("/readyz").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.05)
                else:
                    raise RuntimeError("Server did not become ready")
            if process.poll() is not None:
                logs.seek(0)
                raise RuntimeError(logs.read().decode(errors="replace"))
            yield url
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def percentile(values, fraction):
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


async def counters(client):
    response = await client.get("/metrics")
    response.raise_for_status()
    result = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name in {"batchserve_batch_size_count", "batchserve_batch_size_sum"}:
                result[sample.name] = int(sample.value)
    return result


async def measure(client, payloads, expected, requests, concurrency, allow_overload=False):
    before = await counters(client)

    async def worker(offset):
        results = []
        for idx in range(offset, requests, concurrency):
            vector = idx % len(payloads)
            started = time.perf_counter()
            response = await client.post("/v1/predict", content=payloads[vector], headers=HEADERS)
            elapsed_ms = (time.perf_counter() - started) * 1000
            results.append((vector, response.status_code, response.content, elapsed_ms))
        return results

    started = time.perf_counter()
    workers = await asyncio.gather(*(worker(i) for i in range(concurrency)))
    elapsed = time.perf_counter() - started
    after = await counters(client)

    statuses = Counter()
    latencies = []
    for results in workers:
        for vector, code, body, ms in results:
            statuses[code] += 1
            if code == 200:
                np.testing.assert_allclose(np.frombuffer(body, dtype="<f4"), expected[vector], rtol=2e-4, atol=2e-5)
                latencies.append(ms)
    if not latencies or statuses.keys() - ({200, 503} if allow_overload else {200}):
        raise RuntimeError(f"Unexpected HTTP outcomes: {dict(statuses)}")
    calls = after["batchserve_batch_size_count"] - before["batchserve_batch_size_count"]
    dispatched = after["batchserve_batch_size_sum"] - before["batchserve_batch_size_sum"]
    if dispatched != statuses[200] or not 1 <= calls <= dispatched:
        raise RuntimeError("Runtime counters do not reconcile with successful predictions")
    return {
        "statuses": dict(statuses), "throughput": len(latencies) / elapsed,
        "p50_ms": percentile(latencies, 0.5), "p95_ms": percentile(latencies, 0.95),
        "runtime_calls": calls, "mean_batch": dispatched / calls,
    }


async def run_mode(url, payloads, expected, requests, concurrency, trials):
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=url, limits=limits, timeout=5, trust_env=False) as client:
        await measure(client, payloads, expected, concurrency * 4, concurrency)
        parallel = [await measure(client, payloads, expected, requests, concurrency) for _ in range(trials)]
        low = await measure(client, payloads, expected, 64, 1)
    fields = ("throughput", "p50_ms", "p95_ms", "runtime_calls", "mean_batch")
    summary = {field: round(statistics.median(row[field] for row in parallel), 2) for field in fields}
    return {
        "parallel_medians": summary,
        "parallel_samples": [
            {key: round(value, 2) for key, value in row.items() if key != "statuses"} for row in parallel
        ],
        "one_client": {key: round(low[key], 2) for key in fields},
        "parallel_http_200_per_trial": requests,
    }


async def overload_drill(url, payloads, expected):
    limits = httpx.Limits(max_connections=96, max_keepalive_connections=96)
    async with httpx.AsyncClient(base_url=url, timeout=8, limits=limits, trust_env=False) as client:
        await measure(client, payloads, expected, 16, 4)
        burst = await measure(client, payloads, expected, 384, 96, allow_overload=True)
        if burst["statuses"].get(503, 0) == 0:
            raise RuntimeError("Burst did not exercise overload rejection")
        recovered = await measure(client, payloads, expected, 16, 1)
        metrics = await client.get("/metrics")
        if "batchserve_queue_depth 0.0" not in metrics.text:
            raise RuntimeError("Queue did not drain")
        return {"burst_http_statuses": burst["statuses"], "after_burst_http_statuses": recovered["statuses"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 64 or args.requests < args.concurrency or args.trials < 1:
        parser.error("require 1 to 64 clients, at least one request per client, and at least one trial")

    rng = np.random.default_rng(402)
    vectors = rng.standard_normal((32, FEATURES), dtype=np.float32)
    payloads = [row.astype("<f4", copy=False).tobytes() for row in vectors]
    reference = OnnxModel(1)
    expected = np.concatenate([reference(row[None, :]) for row in vectors])
    output = {"requests": args.requests, "concurrency": args.concurrency, "trials": args.trials}
    for label, batch, wait in (("single", 1, 0), ("dynamic", 16, 2)):
        with server(batch, wait, 128) as url:
            output[label] = asyncio.run(run_mode(url, payloads, expected, args.requests, args.concurrency, args.trials))
    with server(4, 2, 4) as url:
        output["overload_drill"] = asyncio.run(overload_drill(url, payloads, expected))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
