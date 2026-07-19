"""
CNN model definition.

This module is responsible for:
- Building the CNN architecture.
- Compiling the model.
- Saving the trained model.
- Loading a saved model.

Author:
    Operating Systems CNN Project
"""

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import load_model

from config.settings import Settings

from tensorflow.keras.layers import Dropout

class CNNModel:
    """
    CNN Model wrapper.
    """

    def __init__(self) -> None:
        self.model: Model = self._build_model()

    def _build_model(self) -> Model:
        """
        Build the CNN architecture.
        """

        model = tf.keras.Sequential(
            [
                Input(shape=Settings.INPUT_SHAPE),

                Conv2D(
                    filters=32,
                    kernel_size=(3, 3),
                    activation="relu",
                    name="conv2d_1",
                ),

                MaxPooling2D(
                    pool_size=(2, 2),
                    name="max_pool_1",
                ),

                Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    activation="relu",
                    name="conv2d_2",
                ),

                MaxPooling2D(
                    pool_size=(2, 2),
                    name="max_pool_2",
                ),

                Flatten(
                    name="flatten",
                ),

                Dense(
                    128,
                    activation="relu",
                    name="dense_1",
                ),

                Dense(
                    10,
                    activation="softmax",
                    name="output",
                ),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=Settings.LEARNING_RATE
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def summary(self) -> None:
        """
        Print model summary.
        """

        self.model.summary()

    def save(self) -> None:
        """
        Save trained model.
        """

        Path(Settings.MODEL_DIR).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save(Settings.MODEL_PATH)

    def load(self) -> None:
        """
        Load saved model.
        """

        self.model = load_model(Settings.MODEL_PATH)

    def get_model(self) -> Model:
        """
        Return keras model.
        """

        return self.model

    def get_layer_names(self) -> list[str]:
        """
        Return all model layer names.
        """

        return [
            layer.name
            for layer in self.model.layers
        ]