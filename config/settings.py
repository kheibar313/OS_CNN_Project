"""
Global project settings.
"""

from pathlib import Path


class Settings:
    """
    Application configuration.
    """

    # ==========================
    # Project Paths
    # ==========================

    BASE_DIR = Path(__file__).resolve().parent.parent

    DATASET_DIR = BASE_DIR / "data"
    DATASET_PATH = DATASET_DIR / "mnist.npz"

    MODEL_DIR = BASE_DIR / "saved_model"
    MODEL_PATH = MODEL_DIR / "model.keras"

    # ==========================
    # Dataset
    # ==========================

    INPUT_SHAPE = (28, 28, 1)

    NUM_CLASSES = 10

    # ==========================
    # Training
    # ==========================

    EPOCHS = 5

    BATCH_SIZE = 32

    LEARNING_RATE = 0.001

    SHUFFLE = True

    # ==========================
    # Threads
    # ==========================

    MAX_LAYER_THREADS = 7