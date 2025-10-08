"""Storage backends available to Zephon IO."""

from .base import StorageBackend
from .local import LocalFSBackend

__all__ = ["LocalFSBackend", "StorageBackend"]
