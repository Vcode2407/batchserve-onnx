import asyncio
from threading import Event

import numpy as np
import pytest

from batchserve.batcher import Batcher, DeadlineExceeded, InferenceFailed, QueueFull, Unavailable
from batchserve.metrics import Metrics


def features(value):
    return np.array([value], dtype=np.float32)


def deadline(seconds=3):
    return asyncio.get_running_loop().time() + seconds


async def wait_for(condition):
    async with asyncio.timeout(2):
        while not condition():
            await asyncio.sleep(0.001)


def test_full_batch_preserves_request_order():
    async def scenario():
        seen = []

        def infer(rows):
            seen.append(rows[:, 0].tolist())
            return rows * 2

        batcher = Batcher(infer, Metrics(), size=4, queue=8, wait_s=5, guard_s=0)
        try:
            output = await asyncio.gather(*(batcher.submit(features(i), deadline()) for i in range(4)))
            assert seen == [[0, 1, 2, 3]]
            assert [row.item() for row in output] == [0, 2, 4, 6]
        finally:
            await batcher.close()

    asyncio.run(scenario())


def test_deadline_guard_flushes_a_partial_batch():
    async def scenario():
        batcher = Batcher(lambda rows: rows, Metrics(), size=4, queue=8, wait_s=5, guard_s=0.5)
        try:
            assert (await batcher.submit(features(7), deadline(0.8))).item() == 7
        finally:
            await batcher.close()

    asyncio.run(scenario())


def test_queue_rejects_overload_and_cancellation_releases_slot():
    async def scenario():
        running, release = Event(), Event()
        seen = []

        def infer(rows):
            seen.extend(rows[:, 0].tolist())
            running.set()
            assert release.wait(3)
            return rows

        batcher = Batcher(infer, Metrics(), size=1, queue=1, wait_s=0, guard_s=0)
        try:
            first = asyncio.create_task(batcher.submit(features(1), deadline()))
            await wait_for(running.is_set)
            queued = asyncio.create_task(batcher.submit(features(2), deadline()))
            await wait_for(lambda: len(batcher.pending) == 1)
            with pytest.raises(QueueFull):
                await batcher.submit(features(3), deadline())
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            assert not batcher.pending
            replacement = asyncio.create_task(batcher.submit(features(4), deadline()))
            await wait_for(lambda: len(batcher.pending) == 1)
            release.set()
            assert (await first).item() == 1
            assert (await replacement).item() == 4
            assert seen == [1, 4]
        finally:
            release.set()
            await batcher.close()

    asyncio.run(scenario())


def test_queued_and_in_flight_deadlines_are_distinct():
    async def scenario():
        running, release = Event(), Event()
        seen = []

        def infer(rows):
            seen.extend(rows[:, 0].tolist())
            running.set()
            assert release.wait(3)
            return rows

        batcher = Batcher(infer, Metrics(), size=1, queue=2, wait_s=0, guard_s=0)
        try:
            first = asyncio.create_task(batcher.submit(features(1), deadline(0.4)))
            await wait_for(running.is_set)
            with pytest.raises(DeadlineExceeded):
                await batcher.submit(features(2), deadline(0.04))
            assert not batcher.pending
            with pytest.raises(DeadlineExceeded):
                await first
            release.set()
            assert (await batcher.submit(features(3), deadline())).item() == 3
            assert seen == [1, 3]
        finally:
            release.set()
            await batcher.close()

    asyncio.run(scenario())


def test_failed_batch_fails_its_requests_and_worker_recovers():
    async def scenario():
        calls = 0

        def infer(rows):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("injected backend failure")
            return rows

        metrics = Metrics()
        batcher = Batcher(infer, metrics, size=2, queue=4, wait_s=0.02, guard_s=0)
        try:
            results = await asyncio.gather(
                *(batcher.submit(features(i), deadline()) for i in range(2)), return_exceptions=True
            )
            assert all(isinstance(result, InferenceFailed) for result in results)
            assert (await batcher.submit(features(5), deadline())).item() == 5
            assert metrics.registry.get_sample_value("batchserve_backend_errors_total") == 1
        finally:
            await batcher.close()

    asyncio.run(scenario())


def test_shutdown_drains_partial_batch_and_stops_admission():
    async def scenario():
        batcher = Batcher(lambda rows: rows, Metrics(), size=4, queue=8, wait_s=5, guard_s=0)
        accepted = asyncio.create_task(batcher.submit(features(8), deadline()))
        await wait_for(lambda: len(batcher.pending) == 1)
        await batcher.close()
        assert (await accepted).item() == 8
        assert not batcher.ready
        with pytest.raises(Unavailable):
            await batcher.submit(features(9), deadline())

    asyncio.run(scenario())
