"""Cache-aware resolver implementation and helpers."""

from ..utils import compute_file_hash
from .errors import CacheInUseError, PermanentSourceMissing, ShardNotReady
from .manager import CacheManager, CacheStats
from .shared_state import CacheSharedState

__all__ = [
    "CacheInUseError",
    "CacheManager",
    "CacheSharedState",
    "CacheStats",
    "PermanentSourceMissing",
    "ShardNotReady",
    "compute_file_hash",
]
