"""
Thread Scheduler.

Responsible for:

- Registering EpochThreads
- Starting threads
- Waiting for completion
- Monitoring execution state
- Reporting execution summary
"""

from __future__ import annotations

import time
from typing import List

from os_layer.epoch_thread import EpochThread


class Scheduler:
    """
    Coordinates the execution of all EpochThreads.
    """

    def __init__(self) -> None:

        self._threads: List[EpochThread] = []

        self._start_time: float | None = None
        self._end_time: float | None = None

    def register(self, thread: EpochThread) -> None:

        self._threads.append(thread)

    def register_many(
        self,
        threads: List[EpochThread],
    ) -> None:

        self._threads.extend(threads)

    def clear(self) -> None:

        self._threads.clear()

    def start_all(self) -> None:
        """
        Execute every epoch thread sequentially.

        Every epoch has its own Thread object,
        but training order remains deterministic.
        """

        self._start_time = time.perf_counter()

        for thread in self._threads:

            thread.start()

            thread.join()

        self._end_time = time.perf_counter()

    def wait_for_completion(self) -> None:
        """
        Kept for API compatibility.
        """

        return

    @property
    def execution_time(self) -> float:

        if (
            self._start_time is None
            or self._end_time is None
        ):
            return 0.0

        return self._end_time - self._start_time

    @property
    def total_threads(self) -> int:

        return len(self._threads)

    @property
    def finished_threads(self) -> int:

        return sum(
            thread.is_finished
            for thread in self._threads
        )

    @property
    def running_threads(self) -> int:

        return sum(
            thread.is_alive()
            for thread in self._threads
        )

    @property
    def threads(self) -> List[EpochThread]:

        return self._threads.copy()

    def print_summary(self) -> None:

        print("\n" + "=" * 60)
        print("Scheduler Summary")
        print("=" * 60)

        print(f"Registered Threads : {self.total_threads}")
        print(f"Finished Threads   : {self.finished_threads}")
        print(f"Running Threads    : {self.running_threads}")
        print(f"Execution Time     : {self.execution_time:.3f} sec")

        print("=" * 60)