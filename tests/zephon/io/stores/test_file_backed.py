from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from zephon.io.dataset import Dataset
from zephon.io.formats.base import FormatHandler
from zephon.io.protocols import RandomAccessShard
from zephon.io.resolvers.base import ShardResolver
from zephon.io.stores.file_backed import FileBackedDatasetShardView
from zephon.io.types import LocalShardFile, LocalShardRef, ShardFile, ShardLocator


class _FakeShard(RandomAccessShard):
    def __init__(self, rows: list[dict[str, object]]):
        self._rows = rows

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._rows[index]

    def __len__(self) -> int:
        return len(self._rows)

    def close(self) -> None:  # pragma: no cover - trivial
        return None


@dataclass
class _FakeResolver(ShardResolver):
    local: LocalShardRef
    raises_first: bool = False
    _did_raise: bool = False
    resolve_calls: int = 0
    touch_calls: int = 0

    def resolve(self, locator: ShardLocator, *, blocking: bool = True) -> LocalShardRef:
        self.resolve_calls += 1
        if self.raises_first and not self._did_raise:
            self._did_raise = True
            raise FileNotFoundError("evicted")
        return self.local

    def touch(self, locator: ShardLocator) -> None:
        self.touch_calls += 1


@dataclass
class _FakeHandler(FormatHandler):
    kind: str = "fakefmt"
    rows: list[dict[str, object]] = None  # type: ignore[assignment]
    open_calls: int = 0

    def discover(
        self, path: str, storage
    ) -> tuple[
        Mapping[int, int], Mapping[int, Mapping[str, object]]
    ]:  # pragma: no cover
        raise NotImplementedError

    def build_locators(self, dataset: Dataset) -> Mapping[int, ShardLocator]:
        raw = ShardFile(basename="raw.bin", bytes=10, hashes={})
        return {
            0: ShardLocator(
                dataset=dataset.name,
                shard_id=0,
                format=self.kind,
                root=dataset.path or "",
                raw=raw,
            )
        }

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard:
        self.open_calls += 1
        return _FakeShard(self.rows or [{"x": 1}])


def _mk_dataset() -> Dataset:
    return Dataset(
        name="d",
        shard_index={0: 2},
        backend={"kind": "fakefmt", "shards": {0: {}}},
        path="/tmp",
    )


def test_file_backed_dataset_view_caches_shards(tmp_path) -> None:
    ds = _mk_dataset()
    handler = _FakeHandler(rows=[{"x": 1}, {"x": 2}])
    local = LocalShardRef(raw=LocalShardFile(path=tmp_path / "raw.bin", bytes=2))
    resolver = _FakeResolver(local=local)

    view = FileBackedDatasetShardView(
        dataset=ds,
        handler=handler,
        resolver=resolver,
        retry_attempts=2,
        retry_initial_backoff=0.0,
        retry_max_backoff=0.0,
    )

    s1, reused1 = view.open(0)
    assert reused1 is False
    s2, reused2 = view.open(0)
    assert reused2 is True
    assert s1 is s2  # cached per shard id
    assert len(s1) == 2
    payload0, _ = s1[0]
    assert payload0 == {"x": 1}

    # Underlying handler.open_shard only called once due to caching within ResilientShard lifecycle
    # (open happens per __getitem__ in ResilientShard; we can only assert resolver was used)
    assert resolver.resolve_calls >= 1


def test_file_backed_dataset_view_retries_on_eviction(tmp_path) -> None:
    ds = _mk_dataset()
    handler = _FakeHandler(rows=[{"x": 5}])
    local = LocalShardRef(raw=LocalShardFile(path=tmp_path / "raw.bin", bytes=1))
    resolver = _FakeResolver(local=local, raises_first=True)

    view = FileBackedDatasetShardView(
        dataset=ds,
        handler=handler,
        resolver=resolver,
        retry_attempts=3,
        retry_initial_backoff=0.0,
        retry_max_backoff=0.0,
    )
    shard, reused = view.open(0)
    assert reused is False
    row, _ = shard[0]
    assert row == {"x": 5}
    # We should have attempted resolve at least twice because of initial failure
    assert resolver.resolve_calls >= 2
    assert resolver.touch_calls >= 1


def test_file_backed_dataset_view_invalid_shard_raises(tmp_path) -> None:
    ds = _mk_dataset()
    handler = _FakeHandler(rows=[{"x": 1}])
    local = LocalShardRef(raw=LocalShardFile(path=tmp_path / "raw.bin", bytes=1))
    resolver = _FakeResolver(local=local)
    view = FileBackedDatasetShardView(
        dataset=ds,
        handler=handler,
        resolver=resolver,
        retry_attempts=1,
        retry_initial_backoff=0.0,
        retry_max_backoff=0.0,
    )
    try:
        _ = view.open(99)
        assert False, "expected KeyError"
    except KeyError:
        pass


def _mk_locator(tmp_path) -> ShardLocator:
    raw = ShardFile(basename="raw.bin", bytes=1, hashes={})
    return ShardLocator(
        dataset="d",
        shard_id=0,
        format="fakefmt",
        root=str(tmp_path),
        raw=raw,
    )
