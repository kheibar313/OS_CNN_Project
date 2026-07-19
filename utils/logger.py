"""
Simple application logger.

Supports:
- Console logging
- GUI callbacks
- File logging (optional)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable


class Logger:
    """
    Simple logger singleton.
    """

    def __init__(self) -> None:

        self.callbacks: list[Callable[[str], None]] = []

        self.log_file: Path | None = None

    def register_callback(
        self,
        callback: Callable[[str], None],
    ) -> None:
        """
        Register GUI callback.
        """

        self.callbacks.append(callback)

    def set_log_file(
        self,
        path: str,
    ) -> None:
        """
        Enable file logging.
        """

        self.log_file = Path(path)

    def log(
        self,
        message: str,
    ) -> None:
        """
        Log one message.
        """

        timestamp = datetime.now().strftime("%H:%M:%S")

        text = f"[{timestamp}] {message}"

        print(text)

        for callback in self.callbacks:

            callback(text)

        if self.log_file is not None:

            with open(
                self.log_file,
                "a",
                encoding="utf-8",
            ) as file:

                file.write(text + "\n")


logger = Logger()