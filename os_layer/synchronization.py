"""
Synchronization utilities.

Responsible for:

- Protecting the training process.
- Providing a single synchronization interface.
- Managing thread-safe execution.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock


class SynchronizationManager:
    """
    Manages synchronization primitives for the project.
    """

    def __init__(self) -> None:

        self._training_lock = Lock()

    @contextmanager
    def training_context(self):
        """
        Allow only one training thread to access the
        TensorFlow model at a time.
        """

        self._training_lock.acquire()

        try:
            yield

        finally:
            self._training_lock.release()