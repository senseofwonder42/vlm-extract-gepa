"""Run async code from synchronous callers (GEPA's engine, Jupyter cells).

GEPA calls `adapter.evaluate(...)` and `reflection_lm(prompt)` synchronously,
while our OpenRouter client is async. `BackgroundLoop` owns one event loop on
a daemon thread; `run(coro)` submits a coroutine to it and blocks until the
result is ready. Using one long-lived loop (instead of `asyncio.run` per call)
keeps the client's connection pool and semaphore valid across calls, and it
also works inside Jupyter, where a loop is already running in the main thread
and `asyncio.run` would fail.
"""

import asyncio
import threading
from collections.abc import Coroutine
from typing import TypeVar

T = TypeVar("T")


class BackgroundLoop:
    """An asyncio event loop running forever on a daemon thread."""

    def __init__(self) -> None:
        """Start the loop thread."""
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, name="vex-async", daemon=True)
        self._thread.start()

    def run(self, coroutine: Coroutine[object, object, T]) -> T:
        """Run a coroutine on the background loop and wait for its result.

        Args:
            coroutine: Coroutine to run.

        Returns:
            Its result (exceptions are re-raised in the caller's thread).
        """
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    def close(self) -> None:
        """Stop the loop and join its thread."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        self.loop.close()
