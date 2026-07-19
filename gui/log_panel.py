"""
GUI Log Panel.
"""

from __future__ import annotations

import tkinter as tk
from tkinter.scrolledtext import ScrolledText


class LogPanel:
    """
    Log widget.
    """

    def __init__(
        self,
        parent,
    ) -> None:

        self.widget = ScrolledText(
            parent,
            width=70,
            height=12,
            state="disabled",
        )

    def pack(self) -> None:

        self.widget.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10,
        )

    def append(
        self,
        text: str,
    ) -> None:

        self.widget.configure(state="normal")

        self.widget.insert(
            tk.END,
            text + "\n",
        )

        self.widget.see(tk.END)

        self.widget.configure(state="disabled")

    def clear(self) -> None:

        self.widget.configure(state="normal")

        self.widget.delete(
            "1.0",
            tk.END,
        )

        self.widget.configure(state="disabled")