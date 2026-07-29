"""Error types raised by cache-aware resolvers."""


class ShardNotReady(RuntimeError):
    """Raised when a shard is still being prepared and blocking access is disabled."""

    def __init__(self, dataset: str, shard_id: int) -> None:
        super().__init__(f"Shard not ready: dataset={dataset} shard={shard_id}")
        self.dataset = dataset
        self.shard_id = shard_id


class PermanentSourceMissing(FileNotFoundError):
    """Raised when the upstream storage reports a shard is permanently missing."""


class CacheInUseError(RuntimeError):
    """Raised when the cache root is held by an incompatible live session.

    Happens when ``session.json`` records a different fingerprint than the
    current process would compute AND the session's registered owners are
    still alive. Resetting here would clobber an actively-running job, so
    we refuse and surface the conflict to the caller.

    The message includes both fingerprints and a per-dataset diff
    (datasets added/removed, shard count or byte deltas, path changes) so
    operators have somewhere to start beyond comparing opaque hashes.
    Sessions written before the per-dataset summary schema landed fall
    back to a "summary missing" notice instead of a structured diff.
    """

    def __init__(
        self,
        cache_root: str,
        existing_fingerprint: str,
        *,
        current_fingerprint: str | None = None,
        diff_lines: list[str] | None = None,
    ) -> None:
        lines = [
            f"Cache root {cache_root!r} is already in use by an incompatible session.",
            f"  stored fingerprint : {existing_fingerprint}",
        ]
        if current_fingerprint is not None:
            lines.append(f"  current fingerprint: {current_fingerprint}")
        if diff_lines:
            lines.append("Differences (per-dataset summary):")
            lines.extend(diff_lines)
        lines.append("Refusing to reset.")
        super().__init__("\n".join(lines))
        self.cache_root = cache_root
        self.existing_fingerprint = existing_fingerprint
        self.current_fingerprint = current_fingerprint
        self.diff_lines = list(diff_lines or [])


__all__ = ["CacheInUseError", "PermanentSourceMissing", "ShardNotReady"]
