from __future__ import annotations

from dataclasses import dataclass

import pytest

from zephon.io.formats.base import FormatHandler
from zephon.io.protocols import RandomAccessShard, SampleLoadStats
from zephon.io.resolvers.base import ShardResolver
from zephon.io.stores.resilient import ResilientShard
from zephon.io.types import LocalShardFile, LocalShardRef, ShardFile, ShardLocator


class _FakeShard(RandomAccessShard):
    def __init__(self, rows: list[dict[str, object]]):
        self._rows = rows

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._rows[index]

    def __len__(self) -> int:  # pragma: no cover - not relevant
        return len(self._rows)

    def close(self) -> None:  # pragma: no cover - no-op
        return None


@dataclass
class _FakeResolver(ShardResolver):
    local: LocalShardRef
    raises_first: bool = False
    resolve_calls: int = 0
    touch_calls: int = 0
    _raised: bool = False

    def resolve(self, locator: ShardLocator, *, blocking: bool = True) -> LocalShardRef:
        self.resolve_calls += 1
        if self.raises_first and not self._raised:
            self._raised = True
            raise FileNotFoundError("evicted")
        return self.local

    def touch(self, locator: ShardLocator) -> None:
        self.touch_calls += 1


@dataclass
class _FakeHandler(FormatHandler):
    kind: str = "fakefmt"
    rows: list[dict[str, object]] | None = None

    def discover(self, path, storage):  # pragma: no cover - unused
        raise NotImplementedError

    def build_locators(self, dataset):  # pragma: no cover - unused
        raise NotImplementedError

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard:
        return _FakeShard(self.rows or [{"x": 1}])


def _locator() -> ShardLocator:
    return ShardLocator(
        dataset="d",
        shard_id=0,
        format="fakefmt",
        root="/tmp",
        raw=ShardFile(basename="raw.bin", bytes=1, hashes={}),
    )


def test_resilient_shard_bounds_checks_without_resolve(tmp_path) -> None:
    loc = _locator()
    ref = LocalShardRef(raw=LocalShardFile(path=tmp_path / "raw.bin", bytes=1))
    resolver = _FakeResolver(local=ref)
    handler = _FakeHandler(rows=[{"x": 1}])
    shard = ResilientShard(
        locator=loc,
        resolver=resolver,
        handler=handler,
        length=2,
        retry_attempts=2,
        retry_initial_backoff=0.0,
        retry_max_backoff=0.0,
    )
    # Negative index should raise before any resolve/open occurs
    try:
        _ = shard[-1]  # type: ignore[index]
        assert False, "expected IndexError"
    except IndexError:
        pass
    # Index >= length should raise without resolving as well
    try:
        _ = shard[2]
        assert False, "expected IndexError"
    except IndexError:
        pass
    assert resolver.resolve_calls == 0 and resolver.touch_calls == 0


def test_resilient_shard_retries_on_eviction_and_touches(tmp_path) -> None:
    loc = _locator()
    ref = LocalShardRef(raw=LocalShardFile(path=tmp_path / "raw.bin", bytes=2))
    resolver = _FakeResolver(local=ref, raises_first=True)
    handler = _FakeHandler(rows=[{"x": 5}, {"x": 6}])
    shard = ResilientShard(
        locator=loc,
        resolver=resolver,
        handler=handler,
        length=2,
        retry_attempts=3,
        retry_initial_backoff=0.0,
        retry_max_backoff=0.0,
    )
    row, stats = shard[0]
    assert row == {"x": 5}
    assert isinstance(stats, SampleLoadStats)
    assert resolver.resolve_calls >= 2  # retried after initial failure
    assert resolver.touch_calls >= 1


def test_resilient_shard_propagates_indexerror_when_length_unknown(tmp_path) -> None:
    loc = _locator()
    ref = LocalShardRef(raw=LocalShardFile(path=tmp_path / "raw.bin", bytes=1))
    resolver = _FakeResolver(local=ref)
    # Handler with one row; request out of range while length=0 (unknown)
    handler = _FakeHandler(rows=[{"x": 1}])
    shard = ResilientShard(
        locator=loc,
        resolver=resolver,
        handler=handler,
        length=0,  # unknown -> boundary not checked up-front
        retry_attempts=1,
        retry_initial_backoff=0.0,
        retry_max_backoff=0.0,
    )
    with pytest.raises(IndexError):
        _ = shard[10]

    shard.close()  # no-op


def test_resilient_shard_records_load_stats(tmp_path) -> None:
    loc = _locator()
    ref = LocalShardRef(
        raw=LocalShardFile(path=tmp_path / "raw.bin", bytes=1),
        cache_hit=True,
    )
    resolver = _FakeResolver(local=ref)
    handler = _FakeHandler(rows=[{"x": 1}])
    shard = ResilientShard(
        locator=loc,
        resolver=resolver,
        handler=handler,
        length=1,
        retry_attempts=1,
        retry_initial_backoff=0.0,
        retry_max_backoff=0.0,
    )
    row, stats = shard[0]
    assert row == {"x": 1}
    assert stats is not None
    assert stats.cache_hits == 1
    assert stats.cache_misses == 0
    assert stats.retries == 0
    assert stats.resolve_ns >= 0
    assert stats.open_ns >= 0
    assert stats.read_ns >= 0
    assert stats.close_ns >= 0
    assert stats.touch_ns >= 0
