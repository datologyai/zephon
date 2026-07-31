"""User-facing dataset descriptors and detectors."""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from zephon._internal.io.catalog import DatasetHeader, ShardCatalogHandle
from zephon._internal.io.formats import ensure_builtin_formats
from zephon._internal.io.formats.base import FormatHandler, get_format
from zephon._internal.io.index import find_and_load_index
from zephon._internal.io.index.index_types import IndexData
from zephon._internal.io.protocols import RandomAccessShard
from zephon._internal.io.storage import RouterStorageBackend, StorageBackend
from zephon.io.memory import InMemoryShard


@dataclass(frozen=True)
class Dataset:
    """Descriptor for a dataset with index and backend info.

    Instances are created via ``from_path`` (file-backed) or ``from_dict``
    (in-memory/testing). They always provide:

    - ``name``: a user-facing identifier used in mixtures
    - ``backend``: opaque metadata that lets FetchOp build a reader later
    - ``path``: original filesystem path if file-backed, otherwise ``None``

    File-backed datasets also carry a few-KB handle to the node-local shard
    catalog (set by ``from_path``); this is what travels in ctx, not
    ``shard_meta``.

    Shard counts are not stored as a mapping; read them via :meth:`ids` /
    :meth:`counts` / :meth:`total` / :meth:`max_count` / :meth:`shard_count`.

    Backend kinds used by the internal store builder:

    - ``litdata``/``mds``/``jsonl``/``parquet``/``vortex``: ``kind`` and
      ``path`` (no per-shard ``shards`` graph — that lives in the catalog now)
    - ``inmem``: ``kind`` and ``shards`` (``dict[int, InMemoryShard]``)

    Note: this class does not expose any method to fetch rows; IO is delegated
    to an internal shard store owned by the FetchOp.
    """

    name: str
    backend: Mapping[str, object]
    path: str | None = None
    # Node-local shard-catalog handle (file-backed datasets only); set by
    # from_path(), not a constructor argument.
    _catalog_handle: ShardCatalogHandle | None = field(
        default=None, compare=False, init=False
    )
    # _ids/_counts are constructor-seeded discovery output, not a cache: at
    # planning time the catalog doesn't exist yet (preflight builds it once per
    # node, later), so they are the only copy the driver can read. Pickling
    # drops them: by spawn time the catalog exists, and attach() maps pages
    # shared node-wide where pickled arrays would be a private copy per worker
    # (in-memory datasets re-derive from the shipped shards). Reads stay uncached.
    _ids: np.ndarray | None = field(default=None, compare=False, repr=False)
    _counts: np.ndarray | None = field(default=None, compare=False, repr=False)

    def _inmem_shards(self) -> Mapping[int, InMemoryShard]:
        shards = self.backend.get("shards")
        assert self.backend.get("kind") == "inmem" and isinstance(shards, Mapping)
        return shards

    def ids(self) -> np.ndarray:
        """Return the sorted shard ids as an ``int64`` array."""
        if self._ids is not None:
            return self._ids
        if self._catalog_handle is not None:
            return self._catalog_handle.attach().ids()
        return _inmem_ids_counts(self._inmem_shards())[0]

    def counts(self) -> np.ndarray:
        """Return per-shard sample counts (``int64``), aligned with :meth:`ids`."""
        if self._counts is not None:
            return self._counts
        if self._catalog_handle is not None:
            return self._catalog_handle.attach().num_rows()
        return _inmem_ids_counts(self._inmem_shards())[1]

    def raw_bytes(self) -> np.ndarray:
        """Return per-shard byte sizes (``int64``), aligned with :meth:`ids`.

        File-backed datasets read the catalog's on-disk shard sizes; in-memory
        shards size their resident payloads lazily (:attr:`InMemoryShard.raw_bytes`).
        """
        if self._catalog_handle is not None:
            return self._catalog_handle.ensure_attached().raw_bytes()
        shards = self._inmem_shards()
        return np.array([shards[i].raw_bytes for i in sorted(shards)], dtype=np.int64)

    def total(self) -> int:
        """Total sample count across all shards."""
        return int(self.counts().sum())

    def max_count(self) -> int:
        """Largest single-shard sample count (0 when empty)."""
        return int(self.counts().max(initial=0))

    def shard_count(self) -> int:
        """Number of shards."""
        return self.ids().size

    def __len__(self) -> int:
        return self.total()

    def __getstate__(self) -> dict[str, object]:
        # Counts never travel — see the _ids/_counts field comment.
        return {
            "name": self.name,
            "backend": self.backend,
            "path": self.path,
            "_catalog_handle": self._catalog_handle,
        }

    def __deepcopy__(self, memo: dict[int, Any]) -> "Dataset":
        # Share, don't copy: a base-WorkSource deepcopy-clone must not duplicate
        # the (immutable) handle and count arrays.
        memo[id(self)] = self
        return self

    @classmethod
    def from_path(cls, name: str, path: str, *, fmt: str | None = None) -> "Dataset":
        """Construct a file-backed dataset descriptor.

        Performs a *count-only* discovery: it obtains ``shard_id`` + ``num_rows``
        as small numpy arrays (the work source's input) without materializing the
        per-shard ``shard_meta`` graph. The full columnar catalog is built once
        per node later, by the Engine's ``finalize()`` (or lazily by the store
        builder for Engine-less use).

        Args:
            name: Logical dataset name used in mixtures and debugging.
            path: Filesystem directory containing the dataset.
            fmt: Optional explicit format. When ``None``, auto-detects.

        Supported formats:

        - ``litdata`` directories containing ``index.json`` structured with
          ``config`` and ``chunks``
        - ``mds`` directories containing ``index.json`` structured with
          ``shards``
        - ``jsonl`` directories where ``*.jsonl`` files act as shards

        Special URI schemes:

        - ``hf://org/name[@rev]/[config/]split`` is served through the
          HuggingFace backend, which streams parquet shards just-in-time via
          the HuggingFace Datasets Server.

        Returns:
            Dataset: a descriptor populated with shard counts and a catalog
            handle for later IO.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
            ValueError: if ``path`` is not a directory or format unsupported.
        """
        url = urllib.parse.urlparse(path)
        is_remote = bool(url.scheme)

        root_path: Path | None = None
        if not is_remote:
            root_path = Path(path)
            if not root_path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {root_path}")
            if not root_path.is_dir():
                raise ValueError(f"Dataset path must be a directory: {root_path}")
            root_path = root_path.resolve()
            root_str = str(root_path)
        else:
            root_str = path.rstrip("/") or path

        storage = RouterStorageBackend()
        kind = fmt
        if kind is None:
            kind = _auto_detect_format(storage, root_str, root_path)
        if kind is None:
            raise ValueError(f"Unsupported dataset format at path: {root_str}")

        ensure_builtin_formats(required={kind})
        handler = get_format(kind)
        ids, counts = _discover_counts(handler, root_str, storage)
        # ids()/counts() hand these arrays out shared (the work-source cursor
        # gathers from them in place); freeze them so an in-place edit fails loud.
        ids.setflags(write=False)
        counts.setflags(write=False)

        backend = {"kind": kind, "path": root_str}
        header = DatasetHeader(name=name, root=root_str, format=kind, path=root_str)
        handle = ShardCatalogHandle(dataset=header)
        dataset = cls(
            name=name,
            backend=backend,
            path=root_str,
            _ids=ids,
            _counts=counts,
        )
        # Non-init private field: set on the frozen instance post-construction.
        object.__setattr__(dataset, "_catalog_handle", handle)
        return dataset

    @classmethod
    def from_dict(cls, name: str, shards: Mapping[int, InMemoryShard]) -> "Dataset":
        """Construct an in-memory dataset descriptor."""
        norm = {int(sid): shard for sid, shard in shards.items()}
        backend: Mapping[str, object] = {"kind": "inmem", "shards": norm}
        ids, counts = _inmem_ids_counts(norm)
        return cls(name=name, backend=backend, path=None, _ids=ids, _counts=counts)


def _inmem_ids_counts(
    shards: Mapping[int, RandomAccessShard],
) -> tuple[np.ndarray, np.ndarray]:
    """Derive aligned ``(ids, counts)`` from resident in-memory shards."""
    ids = np.array(sorted(int(k) for k in shards), dtype=np.int64)
    counts = np.array([len(shards[int(i)]) for i in ids], dtype=np.int64)
    # Handed out shared (same contract as the from_path arrays): freeze them.
    ids.setflags(write=False)
    counts.setflags(write=False)
    return ids, counts


def _discover_counts(
    handler: FormatHandler, path: str, storage: StorageBackend
) -> tuple[np.ndarray, np.ndarray]:
    """Obtain ``(ids, counts)`` without materializing per-shard ``shard_meta``.

    ``FormatHandler.discover_counts`` gives index formats a metadata-only fast
    path; the protocol default runs the full ``discover()`` and keeps only the
    counts, so the heavy graph is at most transiently built, never retained.
    Results are normalized to ``int64`` arrays.
    """
    ids, counts = handler.discover_counts(path, storage)
    return np.asarray(ids, dtype=np.int64), np.asarray(counts, dtype=np.int64)


__all__ = ["Dataset"]


def _auto_detect_format(
    storage: RouterStorageBackend, root_str: str, root_path: Path | None
) -> str | None:
    result = find_and_load_index(root_str, storage)
    if result is not None:
        return _classify_index_payload(result)

    if root_path is not None:
        entries = [p.name for p in root_path.iterdir() if p.is_file()]
    else:
        try:
            entries = storage.listdir(root_str)
        except Exception:
            entries = []

    if any(name.endswith(".jsonl") for name in entries):
        return "jsonl"
    if any(name.endswith(".vortex") for name in entries):
        return "vortex"
    if any(name.endswith(".parquet") for name in entries):
        return "parquet"
    return None


def _classify_index_payload(data: IndexData) -> str:
    if isinstance(data, dict):
        if "chunks" in data and "config" in data:
            return "litdata"
        if "shards" in data:
            # Could be MDS, Parquet, or Vortex index
            # Check if shards have Parquet-specific fields in extra
            shards = data.get("shards", [])
            if shards:
                first_shard = shards[0] if isinstance(shards, list) else shards.get(0)
                if isinstance(first_shard, dict):
                    extra = first_shard.get("extra", {})
                    if isinstance(extra, dict) and "row_groups" in extra:
                        return "parquet"
            return "mds"
    return "mds"
