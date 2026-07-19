"""
Layer Thread.

Represents one logical CNN layer.
"""

from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Semaphore, Thread
from typing import Callable


class LayerThread(Thread):
    """
    Worker thread representing one CNN layer.
    """

    def __init__(
        self,
        layer_name: str,
        task_queue: Queue,
        scheduler_semaphore: Semaphore,
        shutdown_event: Event,
    ) -> None:

        super().__init__(
            daemon=True,
            name=layer_name,
        )

        self.layer_name = layer_name

        self.task_queue = task_queue

        self.scheduler_semaphore = scheduler_semaphore

        self.shutdown_event = shutdown_event

    def run(self) -> None:
        """
        Main worker loop.
        """

        if meta and self.logger:
            self.logger(
                f"[Epoch {meta['epoch']}] "
                f"[{self.layer_name}] "
                f"→ RUNNING ({meta['layer']})"
            )

        while not self.shutdown_event.is_set():

            self.scheduler_semaphore.acquire()

            if self.shutdown_event.is_set():
                break

            try:
                task: Callable[[], None] = (
                    self.task_queue.get_nowait()
                )

            except Empty:
                continue

            try:
                task()

            finally:
                self.task_queue.task_done()
        
        if meta and self.logger:
            self.logger(
                f"[Epoch {meta['epoch']}] "
                f"[{self.layer_name}] "
                f"→ DONE ({meta['layer']})"
            )