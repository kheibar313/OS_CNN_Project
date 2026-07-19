# """
# Layer Thread.

# Represents one logical CNN layer.
# """

# from __future__ import annotations

# from queue import Empty, Queue
# from threading import Event, Semaphore, Thread
# from typing import Callable


# class LayerThread(Thread):
#     """
#     Worker thread responsible for one CNN layer.
#     """

#     def __init__(
#         self,
#         layer_name: str,
#         shutdown_event: Event,
#     ) -> None:

#         super().__init__(
#             daemon=True,
#             name=layer_name,
#         )

#         self.worker_id = layer_name
#         self.logger = print  # later can replace with GUI logger

#         self.layer_name = layer_name

#         self.shutdown_event = shutdown_event

#         self.task_queue: Queue[Callable[[], None]] = Queue()

#         self.semaphore = Semaphore(0)

#     def submit(self, task: Callable[[], None], meta: dict | None = None) -> None:
#         """
#         Submit one task to this layer.
#         """

#         self.task_queue.put((task, meta))
#         self.semaphore.release()

#     def wait_completion(self) -> None:
#         """
#         Wait until all submitted tasks finish.
#         """

#         self.task_queue.join()

#     def run(self) -> None:
#         """
#         Main worker loop.
#         """

#         task, meta = self.task_queue.get_nowait()
#         if meta:
#             print(f"[Layer={self.layer_name}] Epoch={meta.get('epoch')} Running {meta.get('layer')}")

#         while True:

#             self.semaphore.acquire()

#             if (
#                 self.shutdown_event.is_set()
#                 and self.task_queue.empty()
#             ):
#                 break

#             try:
#                 task = self.task_queue.get_nowait()

#             except Empty:
#                 continue

#             try:
#                 self.logger(f"[{self.name}] Executing task")
#                 task()

#             finally:

#                 self.task_queue.task_done()

"""
Layer Thread.

Represents one logical CNN layer worker.
"""

from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Semaphore, Thread
from typing import Callable


class LayerThread(Thread):

    def __init__(
        self,
        layer_name: str,
        shutdown_event: Event,
        logger=None,
    ) -> None:

        super().__init__(daemon=True, name=layer_name)

        self.layer_name = layer_name
        self.shutdown_event = shutdown_event
        self.logger = logger

        self.task_queue: Queue = Queue()
        self.semaphore = Semaphore(0)

    def submit(self, task: Callable[[], None], meta: dict | None = None) -> None:
        self.task_queue.put((task, meta))
        self.semaphore.release()

    def wait_completion(self) -> None:
        self.task_queue.join()

    def run(self) -> None:

        while True:

            self.semaphore.acquire()

            if self.shutdown_event.is_set() and self.task_queue.empty():
                break

            try:
                task, meta = self.task_queue.get_nowait()
            except Empty:
                continue

            try:
                # ✅ LOG BEFORE EXECUTION
                if meta and self.logger:
                    self.logger(
                        f"[Epoch {meta['epoch']}] "
                        f"[Layer {meta['layer']}] "
                        f"[Worker {self.layer_name}] → RUNNING"
                    )

                task()

                # ✅ LOG AFTER EXECUTION
                if meta and self.logger:
                    self.logger(
                        f"[Epoch {meta['epoch']}] "
                        f"[Layer {meta['layer']}] "
                        f"[Worker {self.layer_name}] → DONE"
                    )

            finally:
                self.task_queue.task_done()