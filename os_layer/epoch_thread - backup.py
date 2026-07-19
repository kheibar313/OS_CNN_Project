# """
# Epoch management thread.

# Each EpochThread executes exactly one training epoch.
# """

# from __future__ import annotations

# from threading import Thread
# from time import perf_counter

# from trainer.epoch_runner import EpochRunner

# from os_layer.synchronization import SynchronizationManager
# from os_layer.thread_status import ThreadStatus


# class EpochThread(Thread):
#     """
#     Thread responsible for executing one epoch.
#     """

#     def __init__(
#         self,
#         epoch_number: int,
#         runner: EpochRunner,
#         synchronization: SynchronizationManager,
#         x_train,
#         y_train,
#         x_test,
#         y_test,
#         batch_size: int,
#     ) -> None:

#         super().__init__(name=f"Epoch-{epoch_number}")

#         self.epoch_number = epoch_number

#         self.runner = runner

#         self.synchronization = synchronization

#         self.x_train = x_train
#         self.y_train = y_train

#         self.x_test = x_test
#         self.y_test = y_test

#         self.batch_size = batch_size

#         self.status = ThreadStatus.CREATED

#         self.started_at: float | None = None
#         self.finished_at: float | None = None

#         self.exception: Exception | None = None

#     def run(self) -> None:
#         """
#         Execute one training epoch.
#         """

#         self.status = ThreadStatus.WAITING

#         with self.synchronization.training_context():

#             self.started_at = perf_counter()

#             self.status = ThreadStatus.RUNNING

#             try:

#                 self.runner.run_epoch(
#                     x_train=self.x_train,
#                     y_train=self.y_train,
#                     x_test=self.x_test,
#                     y_test=self.y_test,
#                     batch_size=self.batch_size,
#                 )

#                 self.status = ThreadStatus.FINISHED

#             except Exception as exc:

#                 self.exception = exc

#                 self.status = ThreadStatus.FAILED

#             finally:

#                 self.finished_at = perf_counter()

#     @property
#     def execution_time(self) -> float | None:

#         if (
#             self.started_at is None
#             or self.finished_at is None
#         ):
#             return None

#         return self.finished_at - self.started_at

#     @property
#     def is_finished(self) -> bool:

#         return self.status == ThreadStatus.FINISHED

#     @property
#     def has_failed(self) -> bool:

#         return self.status == ThreadStatus.FAILED

"""
Epoch management thread.

Each EpochThread executes exactly one training epoch.
"""

from __future__ import annotations

from threading import Thread
from time import perf_counter

from tensorflow.keras.callbacks import History

from trainer.epoch_runner import EpochRunner

from os_layer.synchronization import SynchronizationManager
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

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

        self.batch_size = batch_size

        self.status = ThreadStatus.CREATED

        self.started_at: float | None = None
        self.finished_at: float | None = None

        self.exception: Exception | None = None

        # -------- Results --------

        self.history: History | None = None

        self.loss: float | None = None

        self.accuracy: float | None = None

    def run(self) -> None:
        """
        Execute one training epoch.
        """

        self.status = ThreadStatus.WAITING

        with self.synchronization.training_context():

            self.started_at = perf_counter()

            self.status = ThreadStatus.RUNNING

            try:

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

            except Exception as exc:

                self.exception = exc

                self.status = ThreadStatus.FAILED

            finally:

                self.finished_at = perf_counter()

    @property
    def execution_time(self) -> float | None:
        """
        Execution time in seconds.
        """

        if (
            self.started_at is None
            or self.finished_at is None
        ):
            return None

        return self.finished_at - self.started_at

    @property
    def is_finished(self) -> bool:
        """
        Returns True if the thread finished successfully.
        """

        return self.status == ThreadStatus.FINISHED

    @property
    def has_failed(self) -> bool:
        """
        Returns True if the thread failed.
        """

        return self.status == ThreadStatus.FAILED