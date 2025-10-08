"""Cache-aware resolver implementation and helpers."""

from ..utils import compute_file_hash
from .errors import PermanentSourceMissing, ShardNotReady
from .manager import CacheManager, CacheStats
from .shared_state import CacheEntry, CacheSharedState

__all__ = [
    "CacheEntry",
    "CacheManager",
    "CacheSharedState",
    "CacheStats",
    "PermanentSourceMissing",
    "ShardNotReady",
    "compute_file_hash",
]
