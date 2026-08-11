"""Public, single-process debug utility for inspecting dataset payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from zephon._internal.io.protocols import DatasetShardView as _DatasetShardView
from zephon._internal.io.protocols import RandomAccessShard as _RandomAccessShard
from zephon._internal.io.stores import (
    build_multi_dataset_store as _build_multi_dataset_store,
)
from zephon._internal.io.stores.registry import DatasetStoreRegistry as _DatasetStore
from zephon.io.options import StoreOptions
from zephon.types import SamplePayload

if TYPE_CHECKING:
    from zephon.io.dataset import Dataset


class DatasetInspector:
    """Inspect raw payloads from one dataset without a work source or pipeline.

    This utility is intended for interactive inspection and dataset validation.
    It does not choose sample order, add record metadata, prefetch, batch, or
    run workers. Training pipelines should continue to use
    :class:`zephon.Pipeline`.

    Use inspectors as context managers so opened shard resources are closed
    promptly. An inspector is single-process and not thread-safe.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        io_options: StoreOptions | Mapping[str, Any] | None = None,
    ) -> None:
        self._dataset = dataset
        self._options = StoreOptions.from_any(io_options)
        self._store: _DatasetStore | None = None
        self._view: _DatasetShardView | None = None
        self._shards: dict[int, _RandomAccessShard] = {}
        self._closed = False

    def __enter__(self) -> "DatasetInspector":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def read(self, shard_id: int, sample_index: int) -> SamplePayload:
        """Return one raw payload identified by its shard and local row index.

        Args:
            shard_id: Dataset-local shard identifier, as returned by
                :meth:`Dataset.ids`.
            sample_index: Zero-based row index inside that shard.

        Returns:
            The payload stored by the underlying format. Payloads are normally
            mappings, but binary formats may return arrays or other supported
            payload values.
        """
        return self.read_many(shard_id, [sample_index])[0]

    def read_many(
        self, shard_id: int, sample_indices: list[int]
    ) -> list[SamplePayload]:
        """Return raw payloads for local row indices from one shard.

        The returned list preserves the order of ``sample_indices``. Supplying
        an empty list performs no I/O and returns an empty list.
        """
        self._ensure_open()
        if not sample_indices:
            return []

        shard = self._shard(shard_id)
        indexed_indices = sorted(
            enumerate(int(index) for index in sample_indices), key=lambda item: item[1]
        )
        got = shard.getsamples([index for _, index in indexed_indices])
        rows = got[0] if isinstance(got, tuple) else got
        result: list[SamplePayload | None] = [None] * len(sample_indices)
        for (position, _), row in zip(indexed_indices, rows, strict=True):
            result[position] = cast(SamplePayload, row)
        return cast(list[SamplePayload], result)

    def close(self) -> None:
        """Close every shard opened by this inspector.

        Calling ``close`` more than once is safe. A closed inspector cannot be
        used for further reads.
        """
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        for shard in self._shards.values():
            try:
                shard.close()
            except BaseException as exc:  # noqa: BLE001 - close all opened shards.
                if first_error is None:
                    first_error = exc
        self._shards.clear()
        self._view = None
        if self._store is not None:
            try:
                self._store.close()
            except BaseException as exc:  # noqa: BLE001 - finish local teardown.
                if first_error is None:
                    first_error = exc
            self._store = None
        if first_error is not None:
            raise first_error

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DatasetInspector is closed")

    def _shard(self, shard_id: int) -> _RandomAccessShard:
        normalized_id = int(shard_id)
        existing = self._shards.get(normalized_id)
        if existing is not None:
            return existing

        # Reuse the same catalog, resolver, format handler, and optional on-disk
        # cache as FetchOp. Catalog attachment uses the process's existing or
        # default directory; inspection must not reconfigure that global state.
        if self._view is None:
            store = _build_multi_dataset_store(
                {0: self._dataset}, options=self._options
            )
            self._store = store
            self._view = store.for_dataset(0)
        shard, _ = self._view.open(normalized_id)
        self._shards[normalized_id] = shard
        return shard


__all__ = ["DatasetInspector"]
