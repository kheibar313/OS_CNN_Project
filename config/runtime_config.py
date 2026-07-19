"""
Runtime configuration.

Stores values that can be changed
without modifying Settings.py.
"""

from __future__ import annotations


class RuntimeConfig:
    """
    Runtime configuration.
    """

    def __init__(self) -> None:

        self.layer_workers = 1

    def set_layer_workers(
        self,
        workers: int,
    ) -> None:
        """
        Set workers per layer.
        """

        workers = max(1, int(workers))

        self.layer_workers = workers

    def get_layer_workers(self) -> int:
        """
        Return workers per layer.
        """

        return self.layer_workers


runtime_config = RuntimeConfig()