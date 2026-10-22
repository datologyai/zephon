"""Pipeline shutdown releases cache ownership before returning to the caller."""

import json
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import pytest

from zephon import Pipeline
from zephon._internal.io.resolvers.cache.manager import CacheManager
from zephon._internal.io.resolvers.cache.shared_state import CacheSharedState
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource


@pytest.mark.integration
@pytest.mark.parametrize("runner", ["inline", "threads", "process"])
@pytest.mark.parametrize("early_close", [False, True])
def test_pipeline_releases_fetch_and_prefetch_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: str, early_close: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "shard.jsonl").write_text(
        "\n".join(json.dumps({"text": f"sample {i}"}) for i in range(20))
    )
    cache = tmp_path / "cache"
    dataset = Dataset.from_path("data", str(source))
    work = StaticMixtureWorkSource(
        [dataset],
        mixture=MixtureSpec({"data": 1.0}),
        chunk_size=4,
        seed=1,
        shuffle_shards=False,
    )
    managers: list[CacheManager] = []
    original_init = CacheManager.__init__

    def retain_manager(self: CacheManager, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        managers.append(self)

    monkeypatch.setattr(CacheManager, "__init__", retain_manager)
    pipe = (
        Pipeline(work)
        .prefetch(buffer_size=4)
        .fetch(parallelism=1)
        .batch(microbatch_size=4, drop_last=False)
        .options(
            runner=runner,
            max_workers=2,
            mtp_mode=False,
            io_options={"cache": {"enabled": True, "root": str(cache)}},
        )
    )
    iterator = iter(pipe)
    names: dict[str, str] = {}
    try:
        assert next(iterator)
        path = cache / ".zephon_cache_state" / "session.json"
        names = json.loads(path.read_text())["shm_names"]
        if not early_close:
            list(iterator)
        iterator.close()

        # Retaining the managers prevents GC from satisfying these assertions.
        assert len(managers) >= 2
        assert all(not manager._close_finalizer.alive for manager in managers)
        assert json.loads(path.read_text())["owners"] == {}
        for name in names.values():
            with pytest.raises(FileNotFoundError):
                segment = shared_memory.SharedMemory(name=name, create=False)
                segment.close()
    finally:
        iterator.close()
        for manager in managers:
            manager.close()
        CacheSharedState.unlink_by_names(names)
