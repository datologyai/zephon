# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Vortex native failures cooperate with Zephon's eviction-aware retries."""

from pathlib import Path
from typing import Any

import pytest

vortex = pytest.importorskip("vortex.io", reason="vortex-data not installed")
import vortex

from zephon._internal.io.formats.vortex import VortexFormat
from zephon._internal.io.resolvers.base import ShardResolver
from zephon._internal.io.stores.resilient import ResilientShard
from zephon._internal.io.types import (
    LocalShardFile,
    LocalShardRef,
    ShardFile,
    ShardLocator,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("indices", [[1], [1, 0, 1]])
@pytest.mark.parametrize("evicted", [True, False], ids=["evicted", "corrupt"])
def test_vortex_retries_eviction_but_not_corrupt_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    indices: list[int],
    evicted: bool,
) -> None:
    """Re-resolve an evicted file, but propagate malformed data without retries."""
    path = tmp_path / "rows.vortex"
    rows = [{"value": 0}, {"value": 1}]
    vortex.io.write(vortex.array(rows), str(path))
    data = path.read_bytes()
    locator = ShardLocator(
        dataset="test",
        shard_id=0,
        format="vortex",
        root=str(tmp_path),
        raw=ShardFile(basename=path.name, bytes=len(data), hashes={}),
    )
    ref = LocalShardRef(raw=LocalShardFile(path=path, bytes=len(data)))

    class Resolver(ShardResolver):
        calls = 0

        def resolve(
            self, locator: ShardLocator, *, blocking: bool = True
        ) -> LocalShardRef:
            self.calls += 1
            if self.calls > 1:
                path.write_bytes(data)
            return ref

        def touch(self, locator: ShardLocator) -> None:
            pass

    native_open = vortex.open
    opens = 0

    def open_after_eviction(*args: Any, **kwargs: Any) -> Any:
        nonlocal opens
        opens += 1
        if opens == 1:
            # Simulate the cache changing after resolution but before native open.
            if evicted:
                path.unlink()
            else:
                path.write_bytes(b"not a vortex file")
        return native_open(*args, **kwargs)

    monkeypatch.setattr(vortex, "open", open_after_eviction)
    resolver = Resolver()
    shard = ResilientShard(
        locator=locator,
        resolver=resolver,
        opener=VortexFormat(),
        length=len(rows),
        retry_attempts=2,
        retry_initial_backoff=0,
        retry_max_backoff=0,
    )
    if evicted:
        actual, stats = shard.getsamples(indices)
        assert actual == [rows[index] for index in indices]
        assert all(item.retries == 1 for item in stats)
        assert resolver.calls == opens == 2
    else:
        with pytest.raises(RuntimeError):
            shard.getsamples(indices)
        assert resolver.calls == opens == 1
