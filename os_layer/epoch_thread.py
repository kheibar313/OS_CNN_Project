"""
Epoch management thread.

Each EpochThread executes exactly one training epoch and
coordinates logical layer execution through the ThreadPool.
"""

from __future__ import annotations

from threading import Thread
from time import perf_counter

from tensorflow.keras.callbacks import History

from trainer.epoch_runner import EpochRunner

from os_layer.synchronization import SynchronizationManager
from os_layer.thread_pool import ThreadPool
from os_layer.thread_status import ThreadStatus


class EpochThread(Thread):
    """
    Thread responsible for executing one training epoch.
    """

    def __init__(
        self,
        epoch_number: int,
        runner: EpochRunner,
        synchronization: SynchronizationManager,
        thread_pool: ThreadPool,
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size: int,
    ) -> None:

        super().__init__(name=f"Epoch-{epoch_number}")

        self.epoch_number = epoch_number
        self.runner = runner
        self.synchronization = synchronization
        self.thread_pool = thread_pool

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size

        self.status = ThreadStatus.CREATED

        self.started_at: float | None = None
        self.finished_at: float | None = None

        self.exception: Exception | None = None
        self.history: History | None = None
        self.loss: float | None = None
        self.accuracy: float | None = None

    # ---------------------------------------------------------
    # LOG helper (safe, optional)
    # ---------------------------------------------------------
    def _log(self, msg: str) -> None:
        if hasattr(self.synchronization, "logger") and self.synchronization.logger:
            self.synchronization.logger(msg)

    # ---------------------------------------------------------
    # Layer dispatch
    # ---------------------------------------------------------
    def _dispatch_layer_tasks(self) -> None:

        self._log(f"Epoch {self.epoch_number} → dispatching layers...")

        for i, layer_name in enumerate(self.thread_pool.layer_names, 1):

            def task(name=layer_name):
                pass  # actual TF execution happens in model

            self._log(
                f"Epoch {self.epoch_number} "
                f"→ queueing layer {i}/{len(self.thread_pool.layer_names)} "
                f"({layer_name})"
            )

            self.thread_pool.submit(
                layer_name=layer_name,
                task=task,
                epoch=self.epoch_number,
            )

        self.thread_pool.wait_completion()

    # ---------------------------------------------------------
    # Main run
    # ---------------------------------------------------------
    def run(self) -> None:

        self._log(f"========== Epoch {self.epoch_number} START ==========")

        self.status = ThreadStatus.WAITING

        with self.synchronization.training_context():

            self.started_at = perf_counter()
            self.status = ThreadStatus.RUNNING

            try:

                self._dispatch_layer_tasks()

                self._log(f"Epoch {self.epoch_number} → running training step...")

                (
                    self.history,
                    self.loss,
                    self.accuracy,
                ) = self.runner.run_epoch(
                    x_train=self.x_train,
                    y_train=self.y_train,
                    x_test=self.x_test,
                    y_test=self.y_test,
                    batch_size=self.batch_size,
                )

                self.status = ThreadStatus.FINISHED

                self._log(
                    f"Epoch {self.epoch_number} DONE | "
                    f"loss={self.loss:.4f} | acc={self.accuracy:.4%}"
                )

            except Exception as exc:

                self.exception = exc
                self.status = ThreadStatus.FAILED

                self._log(
                    f"Epoch {self.epoch_number} FAILED → {exc}"
                )

            finally:

                self.finished_at = perf_counter()

                self._log(
                    f"========== Epoch {self.epoch_number} END =========="
                )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    @property
    def execution_time(self) -> float | None:

        if self.started_at is None or self.finished_at is None:
            return None

        return self.finished_at - self.started_at

    @property
    def is_finished(self) -> bool:
        return self.status == ThreadStatus.FINISHED

    @property
    def has_failed(self) -> bool:
        return self.status == ThreadStatus.FAILED