# """
# Thread Pool.

# Responsible for:

# - Creating LayerThreads
# - Starting worker threads
# - Stopping worker threads
# - Dispatching tasks to specific layers
# """

# from __future__ import annotations

# from threading import Event

# from os_layer.layer_thread import LayerThread


# class ThreadPool:
#     """
#     Pool of LayerThreads.
#     """

#     def __init__(
#         self,
#         layer_names: list[str],
#         workers_per_layer: int = 1,
#     ) -> None:

#         self.shutdown_event = Event()

#         self.layers: dict[str, list[LayerThread]] = {}

#         for layer_name in layer_names:

#             workers: list[LayerThread] = []

#             for worker in range(workers_per_layer):

#                 thread = LayerThread(
#                     layer_name=f"{layer_name}-{worker + 1}",
#                     shutdown_event=self.shutdown_event,
#                 )

#                 workers.append(thread)

#             self.layers[layer_name] = workers

#     def start(self) -> None:
#         """
#         Start every LayerThread.
#         """

#         for workers in self.layers.values():

#             for thread in workers:

#                 if not thread.is_alive():

#                     thread.start()

#     def submit(self, layer_name: str, task, epoch: int) -> None:
#         workers = self.layers[layer_name]

#         workers[0].submit(
#             task,
#             meta={
#                 "epoch": epoch,
#                 "layer": layer_name,
#             }
#         )   

#     def wait_completion(self) -> None:
#         """
#         Wait until every submitted task finishes.
#         """

#         for workers in self.layers.values():

#             for thread in workers:

#                 thread.wait_completion()

#     def shutdown(self) -> None:
#         """
#         Stop every LayerThread.
#         """

#         self.shutdown_event.set()

#         for workers in self.layers.values():

#             for thread in workers:

#                 thread.semaphore.release()

#         for workers in self.layers.values():

#             for thread in workers:

#                 thread.join()

#     @property
#     def total_threads(self) -> int:
#         """
#         Return total number of LayerThreads.
#         """

#         return sum(
#             len(workers)
#             for workers in self.layers.values()
#         )

#     @property
#     def layer_names(self) -> list[str]:
#         """
#         Return registered layer names.
#         """

#         return list(self.layers.keys())

"""
Thread Pool.

Responsible for:
- Creating LayerThreads
- Starting worker threads
- Stopping worker threads
- Dispatching tasks with metadata
"""

from __future__ import annotations

from threading import Event
from os_layer.layer_thread import LayerThread


class ThreadPool:
    """
    Pool of LayerThreads.
    """

    def __init__(
        self,
        layer_names: list[str],
        workers_per_layer: int = 1,
        logger=None,   # <-- NEW (for UI logging)
    ) -> None:

        self.shutdown_event = Event()
        self.logger = logger

        self.layers: dict[str, list[LayerThread]] = {}

        for layer_name in layer_names:

            workers: list[LayerThread] = []

            for worker in range(workers_per_layer):

                thread = LayerThread(
                    layer_name=f"{layer_name}-{worker + 1}",
                    shutdown_event=self.shutdown_event,
                    logger=self.logger,
                )

                workers.append(thread)

            self.layers[layer_name] = workers

    def start(self) -> None:
        for workers in self.layers.values():
            for thread in workers:
                if not thread.is_alive():
                    thread.start()

    def submit(self, layer_name: str, task, epoch: int) -> None:
        workers = self.layers[layer_name]
        worker = workers[0]

        worker.submit(
            task,
            meta={
                "epoch": epoch,
                "layer": layer_name,
                "worker": worker.name,
            },
        )

    def wait_completion(self) -> None:
        for workers in self.layers.values():
            for thread in workers:
                thread.wait_completion()

    def shutdown(self) -> None:
        self.shutdown_event.set()

        for workers in self.layers.values():
            for thread in workers:
                thread.semaphore.release()

        for workers in self.layers.values():
            for thread in workers:
                thread.join()

    @property
    def total_threads(self) -> int:
        return sum(len(w) for w in self.layers.values())

    @property
    def layer_names(self) -> list[str]:
        return list(self.layers.keys())