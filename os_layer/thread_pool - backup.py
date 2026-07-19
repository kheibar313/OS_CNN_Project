"""
Thread Pool.

Responsible for:

- Creating LayerThreads
- Starting worker threads
- Stopping worker threads
- Managing shared task queue
"""

from __future__ import annotations

from queue import Queue
from threading import Event, Semaphore

from os_layer.layer_thread import LayerThread


class ThreadPool:
    """
    Pool of LayerThreads.
    """

    def __init__(
        self,
        layer_names: list[str],
        workers_per_layer: int = 1,
    ) -> None:

        self.task_queue: Queue = Queue()

        self.shutdown_event = Event()

        self.scheduler_semaphore = Semaphore(0)

        self.threads: list[LayerThread] = []

        for layer_name in layer_names:

            for worker in range(workers_per_layer):

                thread = LayerThread(
                    layer_name=(
                        f"{layer_name}-{worker + 1}"
                    ),
                    task_queue=self.task_queue,
                    scheduler_semaphore=self.scheduler_semaphore,
                    shutdown_event=self.shutdown_event,
                )

                self.threads.append(thread)

    def start(self) -> None:
        """
        Start every LayerThread.
        """

        for thread in self.threads:

            if not thread.is_alive():

                thread.start()

    def submit(
        self,
        task,
    ) -> None:
        """
        Submit one task.
        """

        self.task_queue.put(task)

        self.scheduler_semaphore.release()

    def wait_completion(self) -> None:
        """
        Wait until every queued task finishes.
        """

        self.task_queue.join()

    def shutdown(self) -> None:
        """
        Stop every worker thread.
        """

        self.shutdown_event.set()

        for _ in self.threads:

            self.scheduler_semaphore.release()

        for thread in self.threads:

            thread.join()

    @property
    def total_threads(self) -> int:
        """
        Number of LayerThreads.
        """

        return len(self.threads)