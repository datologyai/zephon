"""Format-agnostic registry of dataset shard views."""

from collections.abc import Callable, Mapping

from zephon._internal.io.protocols import DatasetShardView, MultiDatasetShardStore


class DatasetStoreRegistry(MultiDatasetShardStore):
    """Simple id->view registry owning their shared resolver lifecycle."""

    def __init__(
        self,
        views: Mapping[int, DatasetShardView] | None = None,
        *,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._views: dict[int, DatasetShardView] = dict(views or {})
        self._on_close = on_close
        self._closed = False

    def register(self, dataset_id: int, view: DatasetShardView) -> None:
        self._views[int(dataset_id)] = view

    def for_dataset(self, dataset_id: int) -> DatasetShardView:
        try:
            return self._views[int(dataset_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset id: {dataset_id}") from exc

    def close(self) -> None:
        """Release resources owned by this store; repeated calls are safe."""
        if self._closed:
            return
        self._closed = True
        self._views.clear()
        if self._on_close is not None:
            self._on_close()


__all__ = ["DatasetStoreRegistry"]
