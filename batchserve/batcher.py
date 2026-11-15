"""Bounded FIFO microbatching on a single asyncio event loop."""

import asyncio
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
import logging

import numpy as np

from .metrics import Metrics

log = logging.getLogger(__name__)


class QueueFull(Exception):
    pass


class DeadlineExceeded(Exception):
    pass


class Unavailable(Exception):
    pass


class InferenceFailed(Exception):
    pass


@dataclass(slots=True)
class Ticket:
    features: np.ndarray
    enqueued: float
    deadline: float
    reply: asyncio.Future


class Batcher:
    def __init__(
        self, infer: Callable[[np.ndarray], np.ndarray], metrics: Metrics,
        *, size: int, queue: int, wait_s: float, guard_s: float,
    ) -> None:
        if not 1 <= size <= 64 or queue < size or wait_s < 0 or guard_s < 0:
            raise ValueError("invalid batch size, queue capacity, or timing")
        self.infer = infer
        self.metrics = metrics
        self.size = size
        self.capacity = queue
        self.wait_s = wait_s
        self.guard_s = guard_s
        self.pending: OrderedDict[asyncio.Future, Ticket] = OrderedDict()
        self.wake = asyncio.Event()
        self.metrics.queued.set_function(lambda: len(self.pending))
        self.closing = False
        self.worker = asyncio.create_task(self._run(), name="batchserve-dispatch")

    @property
    def ready(self) -> bool:
        return not self.closing and not self.worker.done()

    async def submit(self, features: np.ndarray, deadline: float) -> np.ndarray:
        loop = asyncio.get_running_loop()
        now = loop.time()
        if not self.ready:
            raise Unavailable
        if now >= deadline:
            raise DeadlineExceeded
        if len(self.pending) >= self.capacity:
            raise QueueFull
        reply = loop.create_future()
        self.pending[reply] = Ticket(features, now, deadline, reply)
        self.wake.set()
        try:
            async with asyncio.timeout_at(deadline):
                return await reply
        except TimeoutError as exc:
            raise DeadlineExceeded from exc
        finally:
            if self.pending.pop(reply, None) is not None:
                self.wake.set()

    async def close(self) -> None:
        self.closing = True
        self.wake.set()
        await self.worker

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while not self.closing or self.pending:
            self.wake.clear()
            cohort = list(islice(self.pending.values(), self.size))
            if not cohort:
                await self.wake.wait()
                continue
            now = loop.time()
            cutoff = min(
                cohort[0].enqueued + self.wait_s,
                min(ticket.deadline for ticket in cohort) - self.guard_s,
            )
            if not self.closing and len(cohort) < self.size and cutoff > now:
                try:
                    async with asyncio.timeout_at(cutoff):
                        await self.wake.wait()
                except TimeoutError:
                    pass
                continue

            batch = []
            for ticket in cohort:
                del self.pending[ticket.reply]
                if ticket.reply.done():
                    continue
                if now >= ticket.deadline:
                    ticket.reply.set_exception(DeadlineExceeded())
                    continue
                self.metrics.queue_seconds.observe(now - ticket.enqueued)
                batch.append(ticket)
            if batch:
                await self._execute(batch)

    async def _execute(self, batch: list[Ticket]) -> None:
        loop = asyncio.get_running_loop()
        self.metrics.batch_size.observe(len(batch))
        started = loop.time()
        try:
            result = await asyncio.to_thread(self._predict, batch)
        except Exception:
            self.metrics.backend_errors.inc()
            log.exception("Inference batch failed")
            for ticket in batch:
                if not ticket.reply.done():
                    ticket.reply.set_exception(InferenceFailed())
            return
        finally:
            self.metrics.runtime_seconds.observe(loop.time() - started)

        finished = loop.time()
        for ticket, vector in zip(batch, result, strict=True):
            if ticket.reply.done():
                continue
            if finished >= ticket.deadline:
                ticket.reply.set_exception(DeadlineExceeded())
            else:
                ticket.reply.set_result(vector)

    def _predict(self, batch: list[Ticket]) -> np.ndarray:
        rows = np.stack([ticket.features for ticket in batch])
        result = self.infer(rows)
        if result.ndim != 2 or result.shape[0] != len(batch):
            raise ValueError("runtime returned an unexpected batch dimension")
        return result
