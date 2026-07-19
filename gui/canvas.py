"""
Drawing Canvas for MNIST digits.

Allows user to draw a digit and export it as MNIST-style image.
"""

from __future__ import annotations

import tkinter as tk
from PIL import Image, ImageDraw


class DrawCanvas:
    """
    Canvas for freehand digit drawing.
    """

    def __init__(
        self,
        parent,
        size: int = 280,
        brush_size: int = 15,
    ) -> None:

        self.size = size
        self.brush_size = brush_size

        self.canvas = tk.Canvas(
            parent,
            width=size,
            height=size,
            bg="white",
        )

        self.image = Image.new(
            "L",
            (size, size),
            color=255,
        )

        self.draw = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None

        self._bind_events()

    def _bind_events(self) -> None:
        """
        Bind mouse events.
        """

        self.canvas.bind("<Button-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)

    def _start_draw(self, event) -> None:

        self.last_x = event.x
        self.last_y = event.y

    def _draw(self, event) -> None:

        x, y = event.x, event.y

        if self.last_x is not None and self.last_y is not None:

            self.canvas.create_line(
                self.last_x,
                self.last_y,
                x,
                y,
                width=self.brush_size,
                fill="black",
                capstyle=tk.ROUND,
                smooth=True,
            )

            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=0,
                width=self.brush_size,
            )

        self.last_x = x
        self.last_y = y

    def _stop_draw(self, event) -> None:

        self.last_x = None
        self.last_y = None

    def pack(self, **kwargs) -> None:

        self.canvas.pack(**kwargs)

    def clear(self) -> None:
        """
        Clear canvas.
        """

        self.canvas.delete("all")

        self.draw.rectangle(
            [0, 0, self.size, self.size],
            fill=255,
        )

    def get_image(self) -> Image:
        """
        Return PIL image (for predictor).
        """

        return self.image