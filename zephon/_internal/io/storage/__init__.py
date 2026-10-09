"""Storage backends available to Zephon IO."""

from ._utils import is_remote_path
from .azure import AzureBackend
from .base import StorageBackend
from .gcs import GCSBackend
from .hf import HFBackend
from .local import LocalFSBackend
from .router import RouterStorageBackend
from .s3 import S3Backend

__all__ = [
    "AzureBackend",
    "GCSBackend",
    "HFBackend",
    "LocalFSBackend",
    "RouterStorageBackend",
    "S3Backend",
    "StorageBackend",
    "is_remote_path",
]
