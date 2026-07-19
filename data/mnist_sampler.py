"""
MNIST Sampler.

Provides random or indexed samples
from the MNIST test dataset.
"""

from __future__ import annotations

import random

from tensorflow.keras.datasets import mnist


class MNISTSampler:
    """
    Provides MNIST test samples.
    """

    def __init__(self) -> None:

        (_, _), (self.x_test, self.y_test) = mnist.load_data()

    def size(self) -> int:
        """
        Return number of test samples.
        """

        return len(self.x_test)

    def random_sample(
        self,
    ) -> tuple[int, object, int]:
        """
        Return one random sample.
        """

        index = random.randint(
            0,
            self.size() - 1,
        )

        return (
            index,
            self.x_test[index],
            int(self.y_test[index]),
        )

    def sample(
        self,
        index: int,
    ) -> tuple[object, int]:
        """
        Return sample by index.
        """

        if (
            index < 0
            or index >= self.size()
        ):
            raise IndexError(
                "Invalid MNIST index."
            )

        return (
            self.x_test[index],
            int(self.y_test[index]),
        )