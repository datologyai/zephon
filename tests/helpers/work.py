from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from zephon.core.constants import SampleId
from zephon.io import Dataset, InMemoryShard
from zephon.work.base import WorkChunk, WorkSource


def make_inmem_dataset(name: str, rows: list[dict[str, Any]]) -> Dataset:
    """Build a simple single-shard in-memory dataset for tests."""
    shard = InMemoryShard(rows)
    return Dataset.from_dict(name, {0: shard})


@dataclass
class FakeIndexableWorkSource(WorkSource):
    """Minimal indexable WorkSource for unit tests.

    - Streams a single component of SampleIds in deterministic order.
    - Supports __len__ and sample_id_at for indexable pipelines.
    - Exposes datasets_by_id for FetchOp to construct an in-memory store.
    """

    dataset: Dataset
    chunk_size: int = 8

    def __post_init__(self) -> None:
        # Initialize WorkSource base (dataclass doesn't call super()).
        super().__init__()
        # Flatten SampleIds from the dataset's shard counts (single shard assumed)
        # dataset_id is fixed to 0 for these tests.
        self._dataset_id = 0
        ids: list[SampleId] = []
        for shard_id, count in zip(
            self.dataset.ids().tolist(), self.dataset.counts().tolist(), strict=True
        ):
            for i in range(count):
                ids.append((self._dataset_id, shard_id, i))
        self._ids = ids
        self._pos: dict[int, int] = {}  # per-lane cursor

    # Streaming API used by Engine when iterating
    def next_chunk_for(
        self,
        lane: int,
        *,
        worker_id: int = 0,
        workers_per_rank: int = 1,
        canonical_replicas: int = 1,
    ) -> WorkChunk | None:
        pos = self._pos.get(lane, 0)
        if pos >= len(self._ids):
            return None
        end = min(pos + int(self.chunk_size), len(self._ids))
        chunk_ids = self._ids[pos:end]
        self._pos[lane] = end
        return WorkChunk(components={"default": list(chunk_ids)}, seed=123)

    # Engine-bound API (lane clones): delegate to per-lane logic using bound lane.
    def next_chunk(self) -> WorkChunk | None:  # type: ignore[override]
        if self._lane is None:
            raise RuntimeError(
                "WorkSource is not bound to a lane. Use clone_for_lane()."
            )
        lane = int(self._lane)
        return self.next_chunk_for(lane)

    # Random-access API for indexable pipelines
    def __len__(self) -> int:  # type: ignore[override]
        return len(self._ids)

    def sample_id_at(self, index: int) -> SampleId:  # type: ignore[override]
        return self._ids[index]

    def supports_indexing(self) -> bool:  # type: ignore[override]
        return True

    # Engine context & checkpointing
    @property
    def datasets_by_id(self) -> Mapping[int, Dataset]:  # type: ignore[override]
        return {self._dataset_id: self.dataset}

    def component_ids(self) -> Mapping[str, int]:
        return {"default": 0}

    def chunk_size_hint(self) -> int | None:  # type: ignore[override]
        return int(self.chunk_size)

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        return {"version": 1, "pos": dict(self._pos)}

    def load_state_dict(self, state: dict[str, Any]) -> None:  # type: ignore[override]
        if int(state.get("version", 1)) != 1:
            raise RuntimeError("Unsupported FakeIndexableWorkSource state version")
        self._pos = {int(k): int(v) for k, v in state.get("pos", {}).items()}
