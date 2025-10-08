"""Format-agnostic registry of dataset shard views."""

from typing import Mapping

from zephon.io.protocols import DatasetShardView, MultiDatasetShardStore


class DatasetStoreRegistry(MultiDatasetShardStore):
    """Simple id->view registry implementing MultiDatasetShardStore."""

    def __init__(self, views: Mapping[int, DatasetShardView] | None = None) -> None:
        self._views: dict[int, DatasetShardView] = dict(views or {})

    def register(self, dataset_id: int, view: DatasetShardView) -> None:
        self._views[int(dataset_id)] = view

    def for_dataset(self, dataset_id: int) -> DatasetShardView:
        try:
            return self._views[int(dataset_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset id: {dataset_id}") from exc


__all__ = ["DatasetStoreRegistry"]
