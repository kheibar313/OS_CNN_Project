"""
Prediction module.

Responsible for:

- Loading an input image
- Preprocessing to MNIST format
- Running inference
- Returning prediction results
"""

from __future__ import annotations

import os
import time

import numpy as np
from PIL import Image

from config.settings import Settings
from model.model_manager import ModelManager


class Predictor:
    """
    Handles digit prediction from images.
    """

    def __init__(self) -> None:
        """
        Initialize predictor using the shared model manager.
        """

        self.manager = ModelManager()

        self.model = self.manager.get_model()

    def _get_valid_path(
        self,
        image_path: str | None,
    ) -> str:
        """
        Ask the user until a valid image path is provided.
        """

        while True:

            if (
                image_path is not None
                and os.path.exists(image_path)
            ):
                return image_path

            print("\nImage not found.")

            image_path = input(
                "Enter image path: "
            ).strip()

    def _preprocess(
        self,
        image_path: str,
    ) -> np.ndarray:
        """
        Convert an image into MNIST input format.
        """

        image = (
            Image.open(image_path)
            .convert("L")
            .resize(
                (
                    Settings.INPUT_SHAPE[1],
                    Settings.INPUT_SHAPE[0],
                )
            )
        )

        image_array = (
            np.asarray(image, dtype=np.float32)
            / 255.0
        )

        image_array = image_array.reshape(
            1,
            *Settings.INPUT_SHAPE,
        )

        return image_array

    def predict(
        self,
        image_path: str | None = None,
    ) -> tuple[int, float, float]:
        """
        Predict the digit contained in an image.

        Returns
        -------
        tuple
            (
                predicted_digit,
                confidence,
                inference_time_seconds,
            )
        """

        image_path = self._get_valid_path(
            image_path
        )

        processed = self._preprocess(
            image_path
        )

        start = time.perf_counter()

        prediction = self.model.predict(
            processed,
            verbose=0,
        )

        inference_time = (
            time.perf_counter() - start
        )

        predicted_digit = int(
            np.argmax(prediction)
        )

        confidence = float(
            np.max(prediction)
        )

        return (
            predicted_digit,
            confidence,
            inference_time,
        )
    
    def predict_image(self, img: Image.Image):
        """
        Predict directly from PIL image.
        """

        start = time.perf_counter()

        img = img.convert("L")
        img = img.resize(
            (Settings.INPUT_SHAPE[0], Settings.INPUT_SHAPE[1])
        )

        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.reshape(img_array, (1, *Settings.INPUT_SHAPE))

        predictions = self.model.predict(img_array, verbose=0)

        end = time.perf_counter()

        digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        inference_time = end - start

        return digit, confidence, inference_time