"""
Batch manager.

Responsible for:

- Managing mini-batches
- Optional shuffling
- Batch iteration

This class does NOT train the model.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


class BatchManager:
    """
    Creates mini-batches from the training dataset.
    """

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> None:

        self.x_train = x_train
        self.y_train = y_train

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.dataset_size = len(x_train)

    def __len__(self) -> int:
        """
        Number of batches.
        """

        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def batches(
        self,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over all mini-batches.
        """

        indices = np.arange(self.dataset_size)

        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.dataset_size, self.batch_size):

            end = start + self.batch_size

            batch_indices = indices[start:end]

            yield (
                self.x_train[batch_indices],
                self.y_train[batch_indices],
            )