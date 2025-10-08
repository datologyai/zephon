"""Error types raised by cache-aware resolvers."""


class ShardNotReady(RuntimeError):
    """Raised when a shard is still being prepared and blocking access is disabled."""

    def __init__(self, dataset: str, shard_id: int) -> None:
        super().__init__(f"Shard not ready: dataset={dataset} shard={shard_id}")
        self.dataset = dataset
        self.shard_id = shard_id


class PermanentSourceMissing(FileNotFoundError):
    """Raised when the upstream storage reports a shard is permanently missing."""


__all__ = ["PermanentSourceMissing", "ShardNotReady"]
