"""User-facing dataset descriptors and detectors."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from zephon.io.formats import ensure_builtin_formats
from zephon.io.formats.base import get_format
from zephon.io.protocols import RandomAccessShard
from zephon.io.storage import LocalFSBackend


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
    - "mds": {"path": str, "shards": metadata}
    - "jsonl": {"path": str, "shards": metadata}
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

        Supported formats:
        - ``"mds"`` directories containing ``index.json``
        - ``"jsonl"`` directories where ``*.jsonl`` files act as shards

        Returns:
            Dataset: a descriptor populated with shard counts and backend
            metadata for later IO.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
            ValueError: if ``path`` is not a directory or format unsupported.
        """
        # TODO(MaxiBoether): What if this is a cloud path? Right now hard coded to local FS.

        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Dataset path must be a directory: {root}")

        root = root.resolve()

        kind = fmt
        if kind is None:
            if (root / "index.json").is_file():
                kind = "mds"
            elif any(p.suffix == ".jsonl" for p in root.iterdir() if p.is_file()):
                kind = "jsonl"

        if kind not in {"mds", "jsonl"}:
            raise ValueError(f"Unsupported dataset format at path: {root}")

        ensure_builtin_formats()
        handler = get_format(kind)
        storage = LocalFSBackend(
            root=root
        )  # TODO(follow up PR): Implement cloud storage backend.
        shard_index, shard_meta = handler.discover(str(root), storage)
        backend = {"kind": kind, "path": str(root), "shards": shard_meta}
        return cls(name=name, shard_index=shard_index, backend=backend, path=str(root))

    @classmethod
    def from_dict(cls, name: str, shards: Mapping[int, RandomAccessShard]) -> "Dataset":
        """Construct an in-memory dataset descriptor."""
        shard_index = {int(sid): int(len(shard)) for sid, shard in shards.items()}
        backend: Mapping[str, object] = {"kind": "inmem", "shards": dict(shards)}
        return cls(name=name, shard_index=shard_index, backend=backend, path=None)


__all__ = ["Dataset"]
