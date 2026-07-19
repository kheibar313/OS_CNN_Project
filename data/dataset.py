"""
Dataset management for the Operating Systems CNN project.

Responsibilities:
- Load the MNIST dataset from a local file.
- Normalize image data.
- Validate dataset integrity.
- Provide access to training and testing sets.

This module does NOT perform batching.
Batch generation is handled by batch_manager.py.
"""

from pathlib import Path
from typing import Tuple

import numpy as np

from config.settings import Settings


class MNISTDataset:
    """
    Handles loading and preprocessing of the MNIST dataset.
    """

    def __init__(self) -> None:
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

        self.x_test: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

    def load(self) -> None:
        """
        Load the MNIST dataset from the local mnist.npz file.
        """

        dataset_path = Path(Settings.DATASET_PATH)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"MNIST dataset not found:\n{dataset_path}\n\n"
                "Place 'mnist.npz' inside the datasets folder."
            )

        data = np.load(dataset_path)

        self.x_train = data["x_train"]
        self.y_train = data["y_train"]

        self.x_test = data["x_test"]
        self.y_test = data["y_test"]

        self._normalize()
        self._validate()

    def _normalize(self) -> None:
        """
        Normalize image pixel values from [0,255] to [0,1].
        """

        self.x_train = self.x_train.astype(np.float32) / 255.0
        self.x_test = self.x_test.astype(np.float32) / 255.0

        self.x_train = self.x_train.reshape(
            -1,
            *Settings.INPUT_SHAPE,
        )

        self.x_test = self.x_test.reshape(
            -1,
            *Settings.INPUT_SHAPE,
        )

    def _validate(self) -> None:
        """
        Ensure the dataset is loaded correctly.
        """

        assert self.x_train is not None
        assert self.y_train is not None
        assert self.x_test is not None
        assert self.y_test is not None

        assert len(self.x_train) == 60000
        assert len(self.y_train) == 60000

        assert len(self.x_test) == 10000
        assert len(self.y_test) == 10000

        assert self.x_train.shape[1:] == Settings.INPUT_SHAPE
        assert self.x_test.shape[1:] == Settings.INPUT_SHAPE

    def get_train_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the training dataset.
        """

        return self.x_train, self.y_train

    def get_test_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the testing dataset.
        """

        return self.x_test, self.y_test

    def train_size(self) -> int:
        """
        Return the number of training samples.
        """

        return len(self.x_train)

    def test_size(self) -> int:
        """
        Return the number of testing samples.
        """

        return len(self.x_test)