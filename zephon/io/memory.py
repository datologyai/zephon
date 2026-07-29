"""The in-memory shard primitive consumed by ``Dataset.from_dict``."""

from functools import cached_property

from zephon._internal.observability.size_estimator import content_bytes

#: Rows sampled per shard when sizing payloads (keeps raw_bytes cheap regardless
#: of shard size).
_BYTE_SAMPLES = 64


class InMemoryShard:
    """List-backed shard useful for tests and quickstarts."""

    def __init__(self, rows: list[dict[str, object]]):
        self._rows = rows

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._rows[index]

    def __len__(self) -> int:
        return len(self._rows)

    @cached_property
    def raw_bytes(self) -> int:
        """Estimated decoded payload bytes for this shard, sampled and cached."""
        n = len(self._rows)
        if n == 0:
            return 0
        step = max(1, n // _BYTE_SAMPLES)
        sampled = [content_bytes(self._rows[i]) for i in range(0, n, step)]
        return int(sum(sampled) / len(sampled) * n)

    def close(self) -> None:
        return None

    def getsamples(self, indices: list[int]) -> list[dict[str, object]]:
        return [self[i] for i in indices]


__all__ = [
    "InMemoryShard",
]
