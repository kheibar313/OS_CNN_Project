"""
Training metrics.

Responsible for collecting and reporting
training statistics across epochs.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass
class EpochMetrics:
    """
    Metrics for a single epoch.
    """

    epoch: int

    loss: float

    accuracy: float

    execution_time: float


class TrainingMetrics:
    """
    Stores metrics collected during training.
    """

    def __init__(self) -> None:

        self._epochs: list[EpochMetrics] = []

    def add(
        self,
        epoch: int,
        loss: float,
        accuracy: float,
        execution_time: float,
    ) -> None:

        self._epochs.append(
            EpochMetrics(
                epoch=epoch,
                loss=loss,
                accuracy=accuracy,
                execution_time=execution_time,
            )
        )

    @property
    def average_loss(self) -> float:

        if not self._epochs:
            return 0.0

        return mean(e.loss for e in self._epochs)

    @property
    def average_accuracy(self) -> float:

        if not self._epochs:
            return 0.0

        return mean(e.accuracy for e in self._epochs)

    @property
    def average_time(self) -> float:

        if not self._epochs:
            return 0.0

        return mean(e.execution_time for e in self._epochs)

    def print_summary(self) -> None:

        print("\n" + "=" * 60)
        print("Training Metrics")
        print("=" * 60)

        for metric in self._epochs:

            print(
                f"Epoch {metric.epoch:2d} | "
                f"Loss={metric.loss:.4f} | "
                f"Accuracy={metric.accuracy:.4%} | "
                f"Time={metric.execution_time:.3f}s"
            )

        print("-" * 60)

        print(
            f"Average Loss     : {self.average_loss:.4f}"
        )

        print(
            f"Average Accuracy : {self.average_accuracy:.4%}"
        )

        print(
            f"Average Time     : {self.average_time:.3f}s"
        )

        print("=" * 60)

    @property
    def epochs(self) -> list[EpochMetrics]:

        return self._epochs.copy()