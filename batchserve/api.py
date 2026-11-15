"""HTTP API and lifecycle for the inference worker."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import os

from fastapi import FastAPI, HTTPException, Request, Response
import numpy as np
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.requests import ClientDisconnect

from .batcher import Batcher, DeadlineExceeded, InferenceFailed, QueueFull, Unavailable
from .metrics import Metrics
from .model import FEATURES, OnnxModel

CONTENT_TYPE = "application/octet-stream"
INPUT_BYTES = FEATURES * 4


@dataclass(frozen=True)
class Settings:
    batch_size: int = 16
    queue_capacity: int = 128
    max_wait_ms: int = 2
    deadline_guard_ms: int = 2
    intra_op_threads: int = 1

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(**{
            field: int(os.getenv(f"BS_{field.upper()}", getattr(defaults, field)))
            for field in cls.__dataclass_fields__
        })


def create_app(settings: Settings | None = None, load_model=OnnxModel) -> FastAPI:
    config = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not 1 <= config.intra_op_threads <= 16:
            raise ValueError("intra-op threads must be between 1 and 16")
        model = load_model(config.intra_op_threads)
        app.state.metrics = Metrics()
        app.state.batcher = Batcher(
            model, app.state.metrics, size=config.batch_size, queue=config.queue_capacity,
            wait_s=config.max_wait_ms / 1000, guard_s=config.deadline_guard_ms / 1000,
        )
        try:
            yield
        finally:
            await app.state.batcher.close()

    app = FastAPI(title="BatchServe ONNX", lifespan=lifespan)

    @app.get("/healthz")
    async def health():
        return {"status": "alive"}

    @app.get("/readyz")
    async def ready(request: Request):
        if not request.app.state.batcher.ready:
            raise HTTPException(503, "Inference scheduler unavailable")
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    async def prometheus(request: Request):
        return Response(
            generate_latest(request.app.state.metrics.registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    @app.post(
        "/v1/predict",
        response_class=Response,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {CONTENT_TYPE: {"schema": {"type": "string", "format": "binary"}}},
            },
            "parameters": [{
                "in": "header", "name": "X-Timeout-Ms", "required": False,
                "schema": {"type": "integer", "minimum": 1, "maximum": 5000, "default": 1000},
            }],
        },
        responses={200: {"content": {CONTENT_TYPE: {"schema": {"type": "string", "format": "binary"}}}}},
    )
    async def predict(request: Request):
        metrics = request.app.state.metrics
        outcome = "error"
        try:
            started = asyncio.get_running_loop().time()
            deadline = started + timeout_ms(request) / 1000
            async with asyncio.timeout_at(deadline):
                features = await read_features(request)
            output = await request.app.state.batcher.submit(features, deadline)
            outcome = "ok"
            return Response(output.astype("<f4", copy=False).tobytes(), media_type=CONTENT_TYPE)
        except HTTPException:
            outcome = "invalid"
            raise
        except (DeadlineExceeded, TimeoutError):
            outcome = "deadline"
            raise HTTPException(504, "Inference deadline exceeded") from None
        except QueueFull:
            outcome = "overloaded"
            raise HTTPException(503, "Inference queue full", headers={"Retry-After": "1"}) from None
        except Unavailable:
            outcome = "unavailable"
            raise HTTPException(503, "Inference scheduler unavailable") from None
        except InferenceFailed:
            raise HTTPException(502, "Model execution failed") from None
        except (asyncio.CancelledError, ClientDisconnect):
            outcome = "cancelled"
            raise
        finally:
            metrics.requests.labels(outcome).inc()

    return app


def timeout_ms(request: Request) -> int:
    value = request.headers.get("X-Timeout-Ms", "1000")
    if not value.isascii() or not value.isdigit() or len(value) > 4 or not 1 <= int(value) <= 5000:
        raise HTTPException(400, "X-Timeout-Ms must be an integer from 1 to 5000")
    return int(value)


async def read_features(request: Request) -> np.ndarray:
    if request.headers.get("Content-Type", "").partition(";")[0].strip().lower() != CONTENT_TYPE:
        raise HTTPException(415, "Expected application/octet-stream")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > INPUT_BYTES:
            raise HTTPException(413, f"Expected at most {INPUT_BYTES} bytes")
        body.extend(chunk)
    if len(body) != INPUT_BYTES:
        raise HTTPException(400, f"Expected {INPUT_BYTES} bytes")
    values = np.frombuffer(body, dtype="<f4").astype(np.float32, copy=False)
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1000:
        raise HTTPException(400, "Features must be finite and in [-1000, 1000]")
    return values


app = create_app()
