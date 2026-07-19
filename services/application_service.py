"""
Application Service.

Provides a single interface for the GUI.
"""

from __future__ import annotations

from trainer.trainer import Trainer
from model.predictor import Predictor


class ApplicationService:
    """
    Main application service.
    """

    def __init__(self) -> None:

        self.predictor = Predictor()

    def train(self) -> None:
        """
        Train the model.
        """

        trainer = Trainer()

        trainer.run()

    def predict(
        self,
        image_path: str,
    ):
        """
        Predict one image.
        """

        return self.predictor.predict(image_path)