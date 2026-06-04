"""User-facing dataset descriptors and detectors."""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from zephon.io.catalog import DatasetHeader, ShardCatalogHandle
from zephon.io.formats import ensure_builtin_formats
from zephon.io.formats.base import FormatHandler, get_format
from zephon.io.index import find_and_load_index
from zephon.io.index.index_types import IndexData
from zephon.io.protocols import RandomAccessShard
from zephon.io.storage import RouterStorageBackend, StorageBackend


@dataclass(frozen=True)
class Dataset:
    """Descriptor for a dataset with index and backend info.

    Instances are created via ``from_path`` (file-backed) or ``from_dict``
    (in-memory/testing). They always provide:
    - ``name``: a user-facing identifier used in mixtures
    - ``shard_index``: mapping ``shard_id -> sample_count`` (computed eagerly)
    - ``backend``: opaque metadata that lets FetchOp build a reader later
    - ``path``: original filesystem path if file-backed, otherwise ``None``
    - ``catalog_handle``: the few-KB handle to the node-local shard catalog
      (file-backed datasets only); this is what travels in ctx, not ``shard_meta``.

    Backend kinds used by the internal store builder:
    - "litdata"/"mds"/"jsonl"/"parquet"/"vortex": {"kind", "path"} (no per-shard
      ``shards`` graph — that lives in the catalog now)
    - "inmem": {"shards": dict[int, RandomAccessShard]}

    Note: this class does not expose any method to fetch rows; IO is delegated
    to an internal shard store owned by the FetchOp.
    """

    name: str
    shard_index: Mapping[int, int]
    backend: Mapping[str, object]
    path: str | None = None
    catalog_handle: ShardCatalogHandle | None = field(default=None, compare=False)

    def ids(self) -> np.ndarray:
        """Return the sorted shard ids as an ``int64`` array."""
        ids = getattr(self, "_ids", None)
        if ids is not None:
            return ids
        return np.array(sorted(self.shard_index), dtype=np.int64)

    def counts(self) -> np.ndarray:
        """Return per-shard sample counts (``int64``), aligned with :meth:`ids`."""
        counts = getattr(self, "_counts", None)
        if counts is not None:
            return counts
        # ``int(k)``: ids() yields numpy int64; index shard_index with plain ints.
        return np.array([self.shard_index[int(k)] for k in self.ids()], dtype=np.int64)

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
        - ``"litdata"`` directories containing ``index.json`` structured with ``config`` and ``chunks``
        - ``"mds"`` directories containing ``index.json`` structured with ``shards``
        - ``"jsonl"`` directories where ``*.jsonl`` files act as shards

        Special URI schemes:
        - ``"hf://org/name[@rev]/[config/]split"`` is served through the
          :class:`zephon.io.storage.hf.HFBackend`, which streams parquet
          shards just-in-time via the HuggingFace Datasets Server. See
          :func:`zephon.io.storage._hf_uri.parse_hf_uri` for the URI grammar.

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
        # shard_index serves per-shard dict lookups; the _ids/_counts arrays set
        # below back the ids()/counts() fast path the work source consumes.
        shard_index = dict(zip(ids.tolist(), counts.tolist(), strict=True))
        dataset = cls(
            name=name,
            shard_index=shard_index,
            backend=backend,
            path=root_str,
            catalog_handle=handle,
        )
        object.__setattr__(dataset, "_ids", ids)
        object.__setattr__(dataset, "_counts", counts)
        return dataset

    @classmethod
    def from_dict(cls, name: str, shards: Mapping[int, RandomAccessShard]) -> "Dataset":
        """Construct an in-memory dataset descriptor."""
        shard_index = {int(sid): int(len(shard)) for sid, shard in shards.items()}
        backend: Mapping[str, object] = {"kind": "inmem", "shards": dict(shards)}
        return cls(name=name, shard_index=shard_index, backend=backend, path=None)


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
