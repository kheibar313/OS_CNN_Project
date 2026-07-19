"""
Thread status definitions.

Used by all project threads to represent
their current execution state.
"""

from __future__ import annotations
from enum import Enum, auto


class ThreadStatus(Enum):
    """
    Represents the lifecycle of a thread.
    """

    CREATED = auto()
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    FAILED = auto()