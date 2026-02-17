"""User-facing dataset descriptors and detectors."""

import json
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from zephon.io.formats import ensure_builtin_formats
from zephon.io.formats.base import get_format
from zephon.io.protocols import RandomAccessShard
from zephon.io.storage import RouterStorageBackend


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
    - "litdata": {"path": str, "shards": metadata}
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
        - ``"litdata"`` directories containing ``index.json`` structured with ``config`` and ``chunks``
        - ``"mds"`` directories containing ``index.json`` structured with ``shards``
        - ``"jsonl"`` directories where ``*.jsonl`` files act as shards

        Returns:
            Dataset: a descriptor populated with shard counts and backend
            metadata for later IO.

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
        shard_index, shard_meta = handler.discover(root_str, storage)
        backend = {"kind": kind, "path": root_str, "shards": shard_meta}
        return cls(name=name, shard_index=shard_index, backend=backend, path=root_str)

    @classmethod
    def from_dict(cls, name: str, shards: Mapping[int, RandomAccessShard]) -> "Dataset":
        """Construct an in-memory dataset descriptor."""
        shard_index = {int(sid): int(len(shard)) for sid, shard in shards.items()}
        backend: Mapping[str, object] = {"kind": "inmem", "shards": dict(shards)}
        return cls(name=name, shard_index=shard_index, backend=backend, path=None)


__all__ = ["Dataset"]


def _auto_detect_format(
    storage: RouterStorageBackend, root_str: str, root_path: Path | None
) -> str | None:
    index_uri: str
    if root_path is not None:
        index_file = root_path / "index.json"
        if index_file.is_file():
            return _detect_index_format(index_file)
        entries = [p.name for p in root_path.iterdir() if p.is_file()]
        if any(name.endswith(".jsonl") for name in entries):
            return "jsonl"
        if any(name.endswith(".vortex") for name in entries):
            return "vortex"
        if any(name.endswith(".parquet") for name in entries):
            return "parquet"
        index_uri = str(index_file)
    else:
        base = root_str.rstrip("/")
        index_uri = f"{base}/index.json" if base else f"{root_str}/index.json"
        try:
            if storage.exists(index_uri):
                data = _load_index_json(storage, index_uri)
                return _classify_index_payload(data)
        except Exception:
            return None

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

    if storage.exists(index_uri):
        data = _load_index_json(storage, index_uri)
        return _classify_index_payload(data)
    return None


def _detect_index_format(index_path: Path) -> str:
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return "mds"
    return _classify_index_payload(data)


def _load_index_json(storage: RouterStorageBackend, path: str) -> Any:
    with storage.open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _classify_index_payload(data: Any) -> str:
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
