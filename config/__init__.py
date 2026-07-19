"""
Configuration package.

Exports the global Settings class so the rest of the project
can simply import it from the config package.
"""

from .settings import Settings

__all__ = ["Settings"]