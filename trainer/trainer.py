"""
Trainer module.

Responsible for:

- Dataset initialization
- Model initialization
- ThreadPool initialization
- Scheduler initialization
- Epoch thread creation
- Training execution
- Metrics collection
- Model evaluation
- Saving trained model
- Dynamic worker configuration (GUI support)
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

from config.runtime_config import runtime_config


class Trainer:
    """
    Main training controller.
    """

    def __init__(self, trainer=None) -> None:
        self.logger = None

        self.trainer = trainer

        self.dataset = MNISTDataset()

        self.model_wrapper = CNNModel()
        self.model = self.model_wrapper.get_model()

        self.batch_manager: BatchManager | None = None

        self.history: list[History] = []

        self.scheduler = Scheduler()
        self.synchronization = SynchronizationManager()

        self.runner = EpochRunner(self.model)
        self.metrics = TrainingMetrics()

        self.thread_pool = ThreadPool(
            layer_names=self.model_wrapper.get_layer_names(),
            workers_per_layer=runtime_config.get_layer_workers(),
            logger=self.logger
        )

        self._init_thread_pool()

    def set_epochs(self, n: int) -> None:
        self.epochs = n

    # ---------------------------------------------------------
    # ThreadPool handling
    # ---------------------------------------------------------

    def _init_thread_pool(self) -> None:
        """
        Create ThreadPool based on runtime config.
        """

        workers = runtime_config.get_layer_workers()

        self.thread_pool = ThreadPool(
            layer_names=self.model_wrapper.get_layer_names(),
            workers_per_layer=workers,
        )

    def set_worker_threads(self, n: int, logger=None) -> None:
        if self.thread_pool:
            self.thread_pool.shutdown()

        runtime_config.set_layer_workers(n)

        self.thread_pool = ThreadPool(
            layer_names=self.model_wrapper.get_layer_names(),
            workers_per_layer=n,
            logger=logger
        )
        
    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------

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

        self.thread_pool.start()

        print("Dataset loaded.")
        print("CNN model created.")
        print(f"Layer Threads : {self.thread_pool.total_threads}")

        self.scheduler.clear()

        for epoch in range(1, self.epochs + 1):

            thread = EpochThread(
                epoch_number=epoch,
                runner=self.runner,
                synchronization=self.synchronization,
                thread_pool=self.thread_pool,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                batch_size=Settings.BATCH_SIZE,
            )

            self.scheduler.register(thread)

        print(f"Epoch Threads : {self.scheduler.total_threads}")
        print("Initialization completed.\n")

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

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
                self.thread_pool.shutdown()
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

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    def save_model(self) -> None:
        """
        Save trained model.
        """

        self.model_wrapper.save()

        print("\nModel saved to:")
        print(Settings.MODEL_PATH)

    # ---------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------

    def run(self) -> None:
        """
        Execute full pipeline.
        """

        try:
            self.initialize()
            self.train()
            self.evaluate()
            self.save_model()

        finally:
            if self.thread_pool:
                self.thread_pool.shutdown()