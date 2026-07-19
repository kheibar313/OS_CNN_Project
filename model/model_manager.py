"""
Model Manager.

Responsible for:

- Loading trained models
- Saving models
- Providing shared model instance
"""

from pathlib import Path

from tensorflow.keras import Model
from tensorflow.keras.models import load_model

from config.settings import Settings


class ModelManager:
    """
    Shared access to the trained CNN model.
    """

    def __init__(self) -> None:

        self._model: Model | None = None

    def load(self) -> Model:
        """
        Load model if not already loaded.
        """

        if self._model is None:

            if not Path(Settings.MODEL_PATH).exists():

                raise FileNotFoundError(
                    "Trained model not found.\n"
                    "Train the model before prediction."
                )

            self._model = load_model(Settings.MODEL_PATH)

        return self._model

    def get_model(self) -> Model:
        """
        Return loaded model.
        """

        return self.load()

    def is_loaded(self) -> bool:
        """
        Check whether the model is already loaded.
        """

        return self._model is not None