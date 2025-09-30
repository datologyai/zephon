"""User-facing dataset descriptors and detectors.

Datasets in Zephon are lightweight descriptors — not readers. They expose
structural information (name, optional path, shard_index) and backend metadata
that the runtime uses later to construct an internal shard store inside the
FetchOp. A ``Dataset`` never performs IO or opens shards by itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from zephon.io.base import RandomAccessShard


@dataclass(frozen=True)
class Dataset:
    """Descriptor for a dataset with index and backend info.

    Instances are created via ``from_path`` (file-backed) or ``from_dict``
    (in-memory/testing). They always provide:
    - ``name``: a user-facing identifier used in mixtures
    - ``shard_index``: mapping ``shard_id -> sample_count`` (computed eagerly)
    - ``backend``: opaque metadata that lets FetchOp build a reader later
    - ``path``: original filesystem path if file-backed, otherwise ``None``

    Backend kinds used by the internal store builder:
    - "mds": {"path": str}
    - "inmem": {"shards": dict[int, RandomAccessShard]}

    Note: this class does not expose any method to fetch rows; IO is delegated
    to an internal shard store owned by the FetchOp.
    """

    name: str
    shard_index: Mapping[int, int]
    backend: Mapping[str, object]
    path: str | None = None

    def __len__(self) -> int:
        return sum(int(v) for v in self.shard_index.values())

    @classmethod
    def from_path(cls, name: str, path: str, *, fmt: str | None = None) -> "Dataset":
        """Construct a file-backed dataset descriptor.

        Args:
            name: Logical dataset name used in mixtures and debugging.
            path: Filesystem directory containing the dataset.
            fmt: Optional explicit format. When ``None``, auto-detects.

        Supported formats: ``"mds"`` (Mosaic MDS). Detection checks for
        ``index.json`` under ``path``. The method parses ``index.json`` to
        materialize the ``shard_index`` but does not instantiate any reader.

        Returns:
            Dataset: a descriptor with ``backend={"kind": "mds", "path": path}``.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
            ValueError: if ``path`` is not a directory, format is unsupported,
                or the index is missing/invalid.
        """
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Dataset path must be a directory: {root}")

        kind = fmt
        if kind is None:
            # Auto-detect MDS via presence of index.json
            if (root / "index.json").is_file():
                kind = "mds"
        if kind != "mds":
            raise ValueError(f"Unsupported dataset format at path: {root}")

        # Parse MDS index.json to compute shard_index
        # TODO(MaxiBoether): maybe do this lazily in worksource?
        index_path = root / "index.json"
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Missing MDS index: {index_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse MDS index: {index_path}") from exc
        shards = data.get("shards")
        if not isinstance(shards, list):
            raise ValueError("MDS index missing 'shards' list")
        shard_index: dict[int, int] = {}
        for shard_id, entry in enumerate(shards):
            samples = entry.get("samples")
            if samples is None:
                raise ValueError(f"Shard {shard_id} missing 'samples' count")
            try:
                count = int(samples)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Shard {shard_id} has invalid sample count: {samples}"
                ) from exc
            shard_index[shard_id] = count
        backend = {"kind": "mds", "path": str(root)}
        return cls(name=name, shard_index=shard_index, backend=backend, path=str(root))

    @classmethod
    def from_dict(cls, name: str, shards: Mapping[int, RandomAccessShard]) -> "Dataset":
        """Construct an in-memory dataset descriptor.

        Args:
            name: Logical dataset name used in mixtures and debugging.
            shards: Mapping of ``shard_id -> RandomAccessShard`` where each
                shard supports ``__len__`` and ``__getitem__``.

        The returned descriptor includes a ``backend={"kind": "inmem",
        "shards": ...}``, which the runtime uses to build an in-memory store
        inside FetchOp. No copying of rows occurs; the mapping is retained.
        """
        shard_index = {int(sid): int(len(shard)) for sid, shard in shards.items()}
        backend: Mapping[str, object] = {"kind": "inmem", "shards": dict(shards)}
        return cls(name=name, shard_index=shard_index, backend=backend, path=None)


__all__ = ["Dataset"]
