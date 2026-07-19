"""
MNIST Test Panel.

Shows random MNIST sample + prediction + image preview.
"""

from __future__ import annotations

import random
import tkinter as tk

import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image, ImageTk


class MNISTPanel:
    """
    Random MNIST test UI.
    """

    def __init__(self, parent, log_callback=None, predictor=None) -> None:

        self.parent = parent
        self.log = log_callback
        self.predictor = predictor

        self.x = None
        self.y = None

        self._load_dataset()

        self.title = tk.Label(parent, text="MNIST Random Test")
        self.image_label = tk.Label(parent)

        self.btn = tk.Button(
            parent,
            text="Random MNIST Test",
            command=self.run_test,
        )

    def _load_dataset(self) -> None:

        (x_train, y_train), _ = mnist.load_data()

        self.x = x_train
        self.y = y_train

        if self.log:
            self.log("[MNIST] dataset loaded")

    def pack(self) -> None:

        self.title.pack(pady=5)
        self.image_label.pack(pady=5)
        self.btn.pack(pady=5)

    def run_test(self) -> None:

        idx = random.randint(0, len(self.x) - 1)

        img_array = self.x[idx]
        true_label = int(self.y[idx])

        # -------- Show image --------
        img = Image.fromarray(img_array)
        img_resized = img.resize((140, 140))

        tk_img = ImageTk.PhotoImage(img_resized)

        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img

        # -------- Predict --------
        digit, conf, t = self.predictor.predict_image(img)

        msg = (
            f"[MNIST] True: {true_label} | "
            f"Pred: {digit} | Conf: {conf:.4f} | Time: {t:.4f}s"
        )

        if self.log:
            self.log(msg)