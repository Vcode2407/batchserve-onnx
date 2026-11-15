import asyncio
from contextlib import asynccontextmanager
from threading import Event

import httpx
import numpy as np

from batchserve.api import CONTENT_TYPE, Settings, create_app
from batchserve.model import FEATURES, OUTPUTS, OnnxModel, weights


BODY = np.zeros(FEATURES, dtype="<f4").tobytes()
HEADERS = {"Content-Type": CONTENT_TYPE}


@asynccontextmanager
async def api_client(app):
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        yield client


def test_onnx_execution_matches_numpy_and_single_item_inference():
    rng = np.random.default_rng(4)
    inputs = rng.standard_normal((8, FEATURES), dtype=np.float32)
    w1, w2, w3 = weights()
    expected = np.maximum(np.maximum(inputs @ w1, 0) @ w2, 0) @ w3
    model = OnnxModel(1)
    batch = model(inputs)
    assert batch.shape == (8, OUTPUTS)
    np.testing.assert_allclose(batch, expected, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(batch, np.concatenate([model(x[None, :]) for x in inputs]),
                               rtol=2e-4, atol=2e-5)


def test_http_body_validation_metrics_and_real_model():
    async def scenario():
        async with api_client(create_app(Settings(batch_size=1, max_wait_ms=0))) as client:
            assert (await client.get("/healthz")).status_code == 200
            assert (await client.get("/readyz")).status_code == 200
            response = await client.post("/v1/predict", content=BODY, headers=HEADERS)
            assert response.status_code == 200
            assert response.headers["content-type"] == CONTENT_TYPE
            output = np.frombuffer(response.content, dtype="<f4")
            assert output.shape == (OUTPUTS,)
            assert np.all(output == 0)

            assert (await client.post("/v1/predict", content=BODY)).status_code == 415
            assert (await client.post("/v1/predict", content=BODY[:-1], headers=HEADERS)).status_code == 400
            assert (await client.post("/v1/predict", content=BODY + b"x", headers=HEADERS)).status_code == 413
            invalid_timeout = await client.post("/v1/predict", content=BODY, headers=HEADERS | {"X-Timeout-Ms": "0"})
            assert invalid_timeout.status_code == 400
            nonfinite = np.full(FEATURES, np.nan, dtype="<f4").tobytes()
            assert (await client.post("/v1/predict", content=nonfinite, headers=HEADERS)).status_code == 400
            metrics = (await client.get("/metrics")).text
            assert 'batchserve_requests_total{outcome="ok"} 1.0' in metrics
            assert 'batchserve_requests_total{outcome="invalid"} 5.0' in metrics
            assert "batchserve_batch_size_count 1.0" in metrics

    asyncio.run(scenario())


def test_http_returns_deadline_and_overload_as_separate_outcomes():
    async def scenario():
        running, release = Event(), Event()

        def infer(rows):
            running.set()
            assert release.wait(3)
            return rows[:, :OUTPUTS]

        app = create_app(Settings(batch_size=1, queue_capacity=1, max_wait_ms=0), load_model=lambda _: infer)
        async with api_client(app) as client:
            async def post(timeout):
                return await client.post("/v1/predict", content=BODY, headers=HEADERS | {"X-Timeout-Ms": str(timeout)})

            try:
                running_request = asyncio.create_task(post(2000))
                async with asyncio.timeout(1):
                    while not running.is_set():
                        await asyncio.sleep(0.001)
                queued = asyncio.create_task(post(250))
                async with asyncio.timeout(1):
                    while not app.state.batcher.pending:
                        await asyncio.sleep(0.001)
                overloaded = await post(1000)
                assert overloaded.status_code == 503
                assert overloaded.headers["Retry-After"] == "1"
                assert (await queued).status_code == 504
                release.set()
                assert (await running_request).status_code == 200
                metrics = (await client.get("/metrics")).text
                assert 'batchserve_requests_total{outcome="overloaded"} 1.0' in metrics
                assert 'batchserve_requests_total{outcome="deadline"} 1.0' in metrics
            finally:
                release.set()

    asyncio.run(scenario())
