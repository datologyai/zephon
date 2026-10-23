# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from zephon._internal.io.stores.open_shards import OpenShardCache
from zephon._internal.io.types import (
    LocalShardFile,
    LocalShardRef,
    ShardFile,
    ShardLocator,
)


class _Shard:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.closed = False
        self.fail = False

    def __getitem__(self, index: int) -> int:
        return self.getsamples([index])[0]

    def getsamples(self, indices: list[int]) -> list[int]:
        if self.fail:
            raise OSError("read failed")
        return indices

    def __len__(self) -> int:
        return 8

    def close(self) -> None:
        self.closed = True


class _Opener:
    def __init__(self) -> None:
        self.opened: list[_Shard] = []

    def open_shard(self, locator: ShardLocator, local_ref: LocalShardRef) -> _Shard:
        self.opened.append(_Shard(local_ref.raw.path))
        return self.opened[-1]


def _ref(path: Path, dataset: str = "data") -> tuple[ShardLocator, LocalShardRef]:
    locator = ShardLocator(
        dataset=dataset,
        shard_id=0,
        format="jsonl",
        root=str(path.parent),
        raw=ShardFile(basename=path.name, bytes=0, hashes={}),
    )
    return locator, LocalShardRef(raw=LocalShardFile(path=path, bytes=0))


def _write(path: Path, data: bytes = b"one") -> Path:
    path.write_bytes(data)
    return path


def _replace(path: Path, data: bytes) -> None:
    # The shard cache installs a new download with a rename, as here.
    replacement = path.with_suffix(".tmp")
    replacement.write_bytes(data)
    replacement.replace(path)


def test_unchanged_file_reuses_its_reader(tmp_path: Path) -> None:
    opener = _Opener()
    cache = OpenShardCache(opener, 4)
    ref = _ref(_write(tmp_path / "a"))

    for _ in range(3):
        lease = cache.open_shard(*ref)
        assert lease.getsamples([2, 0]) == [2, 0]
        assert lease[1] == 1
        assert len(lease) == 8
        lease.close()
        lease.close()

    assert len(opener.opened) == 1
    assert not opener.opened[0].closed

    cache.close()
    assert opener.opened[0].closed


def test_replaced_file_gets_a_new_reader(tmp_path: Path) -> None:
    opener = _Opener()
    cache = OpenShardCache(opener, 4)
    path = _write(tmp_path / "a")
    cache.open_shard(*_ref(path)).close()

    _replace(path, b"other")
    cache.open_shard(*_ref(path)).close()

    assert len(opener.opened) == 2
    assert [shard.closed for shard in opener.opened] == [True, False]


def test_removed_file_raises_and_closes_its_reader(tmp_path: Path) -> None:
    opener = _Opener()
    cache = OpenShardCache(opener, 4)
    path = _write(tmp_path / "a")
    cache.open_shard(*_ref(path)).close()

    path.unlink()
    with pytest.raises(FileNotFoundError):
        cache.open_shard(*_ref(path))

    assert opener.opened[0].closed


def test_new_reader_closes_readers_of_removed_files(tmp_path: Path) -> None:
    opener = _Opener()
    cache = OpenShardCache(opener, 4)
    removed = _write(tmp_path / "a")
    cache.open_shard(*_ref(removed)).close()

    # Only a new reader looks at other files, so their disk space is freed then.
    removed.unlink()
    cache.open_shard(*_ref(_write(tmp_path / "b"))).close()

    assert [shard.closed for shard in opener.opened] == [True, False]


def test_least_recently_used_reader_closes_above_the_limit(tmp_path: Path) -> None:
    opener = _Opener()
    cache = OpenShardCache(opener, 2)
    paths = [_write(tmp_path / name) for name in ("a", "b", "c")]
    cache.open_shard(*_ref(paths[0])).close()
    cache.open_shard(*_ref(paths[1])).close()
    cache.open_shard(*_ref(paths[0])).close()
    cache.open_shard(*_ref(paths[2])).close()

    assert [shard.path for shard in opener.opened] == paths
    assert [shard.closed for shard in opener.opened] == [False, True, False]


def test_evicted_reader_closes_with_its_last_lease(tmp_path: Path) -> None:
    opener = _Opener()
    cache = OpenShardCache(opener, 1)
    first = cache.open_shard(*_ref(_write(tmp_path / "a")))
    cache.open_shard(*_ref(_write(tmp_path / "b"))).close()

    assert not opener.opened[0].closed
    assert first.getsamples([3]) == [3]
    first.close()
    assert opener.opened[0].closed


def test_failed_read_drops_the_reader(tmp_path: Path) -> None:
    opener = _Opener()
    cache = OpenShardCache(opener, 4)
    ref = _ref(_write(tmp_path / "a"))
    lease = cache.open_shard(*ref)
    opener.opened[0].fail = True

    with pytest.raises(OSError, match="read failed"):
        lease.getsamples([0])
    assert not opener.opened[0].closed
    lease.close()

    assert opener.opened[0].closed
    assert cache.open_shard(*ref).getsamples([1]) == [1]
    assert len(opener.opened) == 2


def test_close_waits_for_open_leases(tmp_path: Path) -> None:
    opener = _Opener()
    cache = OpenShardCache(opener, 4)
    cache.open_shard(*_ref(_write(tmp_path / "a"))).close()
    leased = cache.open_shard(*_ref(_write(tmp_path / "b")))

    cache.close()
    assert [shard.closed for shard in opener.opened] == [True, False]
    leased.close()
    assert opener.opened[1].closed


def test_datasets_sharing_a_file_get_separate_readers(tmp_path: Path) -> None:
    """Datasets that share a file can read it differently, as Lance versions do."""
    opener = _Opener()
    cache = OpenShardCache(opener, 4)
    path = _write(tmp_path / "a")
    cache.open_shard(*_ref(path, "v1")).close()
    cache.open_shard(*_ref(path, "v2")).close()

    assert len(opener.opened) == 2


def test_max_shards_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        OpenShardCache(_Opener(), 0)
