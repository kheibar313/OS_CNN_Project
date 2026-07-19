# """
# Trainer module.

# Responsible for:

# - Dataset initialization
# - Model initialization
# - Scheduler initialization
# - Epoch thread creation
# - Training execution
# - Model evaluation
# - Saving trained model
# """

# from __future__ import annotations

# from tensorflow.keras.callbacks import History

# from config.settings import Settings
# from data.batch_manager import BatchManager
# from data.dataset import MNISTDataset
# from model.cnn_model import CNNModel

# from trainer.epoch_runner import EpochRunner

# from os_layer.scheduler import Scheduler
# from os_layer.epoch_thread import EpochThread
# from os_layer.synchronization import SynchronizationManager


# class Trainer:
#     """
#     Main training controller.
#     """

#     def __init__(self) -> None:

#         self.dataset = MNISTDataset()

#         self.model_wrapper = CNNModel()

#         self.model = self.model_wrapper.get_model()

#         self.batch_manager: BatchManager | None = None

#         self.history: list[History] = []

#         self.scheduler = Scheduler()

#         self.synchronization = SynchronizationManager()

#         self.runner = EpochRunner(self.model)

#     def initialize(self) -> None:
#         """
#         Initialize every required component.
#         """

#         print("=" * 60)
#         print("Initializing trainer...")
#         print("=" * 60)

#         self.dataset.load()

#         x_train, y_train = self.dataset.get_train_data()

#         x_test, y_test = self.dataset.get_test_data()

#         self.batch_manager = BatchManager(
#             x_train=x_train,
#             y_train=y_train,
#             batch_size=Settings.BATCH_SIZE,
#             shuffle=Settings.SHUFFLE,
#         )

#         print("Dataset loaded.")

#         print("CNN model created.")

#         self.scheduler.clear()

#         for epoch in range(1, Settings.EPOCHS + 1):

#             thread = EpochThread(
#                 epoch_number=epoch,
#                 runner=self.runner,
#                 synchronization=self.synchronization,
#                 x_train=x_train,
#                 y_train=y_train,
#                 x_test=x_test,
#                 y_test=y_test,
#                 batch_size=Settings.BATCH_SIZE,
#             )

#             self.scheduler.register(thread)

#         print(
#             f"{self.scheduler.total_threads} "
#             "EpochThreads created."
#         )

#         print("Initialization completed.\n")

#     def train(self) -> None:
#         """
#         Execute training using EpochThreads.
#         """

#         print("=" * 60)
#         print("Training started...")
#         print("=" * 60)

#         self.scheduler.start_all()

#         self.scheduler.wait_for_completion()

#         print("\nTraining finished.")

#         self.scheduler.print_summary()

#     def evaluate(self) -> tuple[float, float]:
#         """
#         Evaluate trained model.
#         """

#         x_test, y_test = self.dataset.get_test_data()

#         loss, accuracy = self.model.evaluate(
#             x_test,
#             y_test,
#             verbose=0,
#         )

#         print("=" * 60)
#         print(f"Test Loss     : {loss:.4f}")
#         print(f"Test Accuracy : {accuracy:.4%}")
#         print("=" * 60)

#         return loss, accuracy

#     def save_model(self) -> None:
#         """
#         Save trained model.
#         """

#         self.model_wrapper.save()

#         print(
#             "\nModel saved to:\n"
#             f"{Settings.MODEL_PATH}"
#         )

#     def run(self) -> None:
#         """
#         Execute the complete training pipeline.
#         """

#         self.initialize()

#         self.train()

#         self.evaluate()

#         self.save_model()

"""
Trainer module.

Responsible for:

- Dataset initialization
- Model initialization
- Layer ThreadPool initialization
- Scheduler initialization
- Epoch thread creation
- Training execution
- Metrics collection
- Model evaluation
- Saving trained model
"""

from __future__ import annotations

from tensorflow.keras.callbacks import History

from config.settings import Settings
from data.batch_manager import BatchManager
from data.dataset import MNISTDataset
from model.cnn_model import CNNModel

from trainer.epoch_runner import EpochRunner
from trainer.metrics import TrainingMetrics

from os_layer.scheduler import Scheduler
from os_layer.epoch_thread import EpochThread
from os_layer.synchronization import SynchronizationManager
from os_layer.thread_pool import ThreadPool


class Trainer:
    """
    Main training controller.
    """

    def __init__(self) -> None:

        self.dataset = MNISTDataset()

        self.model_wrapper = CNNModel()

        self.model = self.model_wrapper.get_model()

        self.batch_manager: BatchManager | None = None

        self.history: list[History] = []

        self.scheduler = Scheduler()

        self.synchronization = SynchronizationManager()

        self.runner = EpochRunner(self.model)

        self.metrics = TrainingMetrics()

        self.thread_pool: ThreadPool | None = None

    def initialize(self) -> None:
        """
        Initialize every required component.
        """

        print("=" * 60)
        print("Initializing trainer...")
        print("=" * 60)

        self.dataset.load()

        x_train, y_train = self.dataset.get_train_data()

        x_test, y_test = self.dataset.get_test_data()

        self.batch_manager = BatchManager(
            x_train=x_train,
            y_train=y_train,
            batch_size=Settings.BATCH_SIZE,
            shuffle=Settings.SHUFFLE,
        )

        print("Dataset loaded.")
        print("CNN model created.")

        # -----------------------------------------
        # Create Layer ThreadPool
        # -----------------------------------------

        layer_names = self.model_wrapper.get_layer_names()

        self.thread_pool = ThreadPool(
            layer_names=layer_names,
            workers_per_layer=1,
        )

        self.thread_pool.start()

        print(
            f"{self.thread_pool.total_threads} LayerThreads created."
        )

        # -----------------------------------------
        # Create Epoch Threads
        # -----------------------------------------

        self.scheduler.clear()

        for epoch in range(1, Settings.EPOCHS + 1):

            thread = EpochThread(
                epoch_number=epoch,
                runner=self.runner,
                synchronization=self.synchronization,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                batch_size=Settings.BATCH_SIZE,
            )

            self.scheduler.register(thread)

        print(
            f"{self.scheduler.total_threads} EpochThreads created."
        )

        print("Initialization completed.\n")

    def train(self) -> None:
        """
        Execute training using EpochThreads.
        """

        print("=" * 60)
        print("Training started...")
        print("=" * 60)

        self.scheduler.start_all()

        self.scheduler.wait_for_completion()

        for thread in self.scheduler.threads:

            if thread.has_failed:

                raise RuntimeError(
                    f"Epoch {thread.epoch_number} failed."
                ) from thread.exception

            self.history.append(thread.history)

            self.metrics.add(
                epoch=thread.epoch_number,
                loss=thread.loss,
                accuracy=thread.accuracy,
                execution_time=thread.execution_time,
            )

        print("\nTraining finished.")

        self.scheduler.print_summary()

        self.metrics.print_summary()

    def evaluate(self) -> tuple[float, float]:
        """
        Evaluate trained model.
        """

        x_test, y_test = self.dataset.get_test_data()

        loss, accuracy = self.model.evaluate(
            x_test,
            y_test,
            verbose=0,
        )

        print("=" * 60)
        print(f"Test Loss     : {loss:.4f}")
        print(f"Test Accuracy : {accuracy:.4%}")
        print("=" * 60)

        return loss, accuracy

    def save_model(self) -> None:
        """
        Save trained model.
        """

        self.model_wrapper.save()

        print(
            "\nModel saved to:\n"
            f"{Settings.MODEL_PATH}"
        )

    def shutdown(self) -> None:
        """
        Shutdown every background thread.
        """

        if self.thread_pool is not None:

            self.thread_pool.shutdown()

    def run(self) -> None:
        """
        Execute the complete training pipeline.
        """

        try:

            self.initialize()

            self.train()

            self.evaluate()

            self.save_model()

        finally:

            self.shutdown()