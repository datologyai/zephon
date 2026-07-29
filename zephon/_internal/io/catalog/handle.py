# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""The travelling handle, the process registry, and ``finalize``/``attach``.

A ``ShardCatalogHandle`` pickles to a few KB (only the content ``fingerprint``,
no buffers or path); the mmap-backed columns live in a module registry that
pickle never touches. ``finalize()`` builds-or-loads under a source-key lock when
the fingerprint is unknown (one build per node) and bakes it; ``attach()`` loads
by the known fingerprint (registry -> node-local file -> cross-node rebuild,
the rebuild serialized on the same source-key lock).
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock

from zephon._internal.io.catalog import io as catalog_io
from zephon._internal.io.catalog.builder import DatasetHeader, build_catalog
from zephon._internal.io.catalog.catalog import ShardCatalog
from zephon._internal.io.catalog.io import SCHEMA_VERSION
from zephon._internal.io.index.index_reader import INDEX_FILENAMES

if TYPE_CHECKING:
    from zephon.io.options import StoreOptions

logger = logging.getLogger(__name__)

# Subdirectory under a local ``cache.root`` where catalogs are co-located. The
# CacheManager must preserve this during a cache wipe (zephon/io/resolvers/cache/
# manager.py) — the catalog is built once per node and is unrelated to the
# cache's session lifecycle.
CATALOG_CACHE_SUBDIR = ".catalog"

# Module globals. Never pickled.
_REGISTRY: dict[tuple[int, str], ShardCatalog] = {}
_REGISTRY_LOCK = threading.Lock()
_CATALOG_DIR: Path | None = None
_CATALOG_DIR_LOCK = threading.Lock()

_NETWORK_FS = {
    "nfs",
    "nfs4",
    "cifs",
    "smb",
    "smbfs",
    "lustre",
    "gpfs",
    "ceph",
    "beegfs",
    "9p",
    "afs",
    "glusterfs",
}
_RAM_FS = {"tmpfs", "ramfs", "devtmpfs"}


class CatalogFingerprintMismatch(RuntimeError):
    """A cross-node rebuild produced a fingerprint other than the baked one.

    For ``runner="remote"`` this is a real correctness failure: the actor's
    ``shard_id -> file`` mapping disagrees with the work the driver assigned.
    """


@dataclass
class ShardCatalogHandle:
    """The few-KB object that travels in ctx; resolves to a shared catalog."""

    dataset: DatasetHeader
    fingerprint: str | None = None

    def finalize(self) -> str:
        return finalize(self)

    def attach(self) -> ShardCatalog:
        return attach(self)

    def ensure_attached(self) -> ShardCatalog:
        """Finalize (idempotent: build-or-load once) and attach in one step."""
        finalize(self)
        return attach(self)


def _fs_type(path: Path) -> str | None:
    """Filesystem type of the mount containing ``path``.

    Returns the fstype string; ``None`` when the platform cannot detect it at all
    (no ``/proc/self/mountinfo``, e.g. macOS); or ``""`` when procfs exists but the
    mount for ``path`` could not be determined (a Linux parse/lookup failure).
    """
    try:
        target = path
        while not target.exists() and target != target.parent:
            target = target.parent
        resolved = str(target.resolve())
    except OSError:
        return None

    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.exists():
        return None  # platform cannot detect (e.g. macOS)
    best_mount = ""
    best_fstype = ""  # procfs present but mount undetermined
    try:
        for line in mountinfo.read_text().splitlines():
            fields = line.split()
            try:
                sep = fields.index("-")
            except ValueError:
                continue
            mount_point = fields[4]
            fstype = fields[sep + 1] if sep + 1 < len(fields) else ""
            if resolved == mount_point or resolved.startswith(
                mount_point.rstrip("/") + "/"
            ):
                if len(mount_point) >= len(best_mount):
                    best_mount = mount_point
                    best_fstype = fstype
    except OSError:
        return ""
    return best_fstype


def _is_network_or_ram(path: Path) -> bool:
    fstype = _fs_type(path)
    if fstype is None:
        return False  # platform can't detect (macOS): assume node-local disk
    if fstype == "":
        # Linux but the mount is undetermined: fail safe (could be network FS).
        return True
    base = fstype.split(".")[0].lower()
    return base in _NETWORK_FS or base.startswith("fuse") or base in _RAM_FS


def _default_tmp_dir() -> Path:
    uid = getattr(os, "getuid", lambda: 0)()
    return Path(tempfile.gettempdir()) / f"zephon-{uid}" / "catalog"


def resolve_catalog_dir(io_options: StoreOptions | None) -> Path:
    """Resolve the node-local catalog directory from this process's config.

    Chain (first hit wins): local ``{cache.root}/.catalog`` when the cache is on,
    else ``$ZEPHON_CATALOG_DIR`` (operator escape hatch), else
    ``{tempdir}/zephon-{uid}/catalog``. The auto-picked cache root is skipped if
    it looks network- or RAM-backed (mmap page-sharing degrades there).
    """
    cache = io_options.cache if io_options is not None else None
    if cache is not None and cache.enabled:
        root = Path(str(cache.root)).expanduser()
        candidate = root / CATALOG_CACHE_SUBDIR
        if not _is_network_or_ram(root):
            return candidate
        logger.warning(
            "Catalog: cache.root %s looks network/RAM-backed; not co-locating "
            "the catalog there.",
            root,
        )

    env = os.environ.get("ZEPHON_CATALOG_DIR")
    if env:
        return Path(env).expanduser()

    tmp = _default_tmp_dir()
    if _is_network_or_ram(tmp):
        logger.warning(
            "Catalog: falling back to %s which looks RAM/network-backed; set "
            "ZEPHON_CATALOG_DIR to a node-local disk path to avoid memory "
            "pressure.",
            tmp,
        )
    return tmp


def set_catalog_dir(io_options: StoreOptions | None) -> Path:
    """Set the process-global catalog directory from ``io_options`` and return it."""
    global _CATALOG_DIR
    resolved = resolve_catalog_dir(io_options)
    with _CATALOG_DIR_LOCK:
        _CATALOG_DIR = resolved
    return resolved


def _catalog_dir() -> Path:
    with _CATALOG_DIR_LOCK:
        if _CATALOG_DIR is not None:
            return _CATALOG_DIR
    # No process-global set yet: fall back to the env/tmp default so direct/test
    # use works without an Engine.
    return resolve_catalog_dir(None)


def _schema_dir(base: Path) -> Path:
    return base / f"v{SCHEMA_VERSION}"


def _source_sig(header: DatasetHeader) -> str:
    """Cheap freshness signature of the discovery inputs (across-run safety net).

    Datasets are assumed immutable during a run; this only protects against a
    dataset changed in place under the same root between runs (a new sig yields a
    new source key -> pointer miss -> rebuild, so a stale catalog is not served).
    """
    # Go through the storage backend (not raw Path), so remote roots (s3://,
    # gs://) are covered too: stat returns {size, mtime} for local and obstore
    # alike, and an in-place overwrite bumps last_modified. RouterStorageBackend
    # is imported lazily to avoid the storage -> catalog import cycle.
    from zephon._internal.io.storage import RouterStorageBackend

    parts = [header.format, header.root]
    try:
        storage = RouterStorageBackend()
        # Reuse discovery's candidate order; the matched name is folded into the
        # sig so an index.json <-> _index.json swap re-keys too.
        for name in INDEX_FILENAMES:
            try:
                st = storage.stat(os.path.join(header.root, name))
            except Exception:
                continue
            parts.append(f"{name}:{st.get('size', '')}:{st.get('mtime', '')}")
            break
        else:
            # No index (scan formats): fold a sorted (name,size) listing so an
            # in-place change re-keys the build.
            for name in sorted(storage.listdir(header.root)):
                try:
                    st = storage.stat(os.path.join(header.root, name))
                    parts.append(f"{name}:{st.get('size', '')}")
                except Exception:
                    parts.append(name)
    except Exception:
        pass
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[
        :16
    ]


def _source_key(header: DatasetHeader) -> str:
    raw = "|".join(
        [str(SCHEMA_VERSION), header.format, header.root, _source_sig(header)]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ensure_catalog_file(
    header: DatasetHeader, *, expected_fp: str | None = None
) -> str:
    """Ensure the content-addressed catalog file exists; return its fingerprint.

    ``finalize()`` and ``attach()``'s cross-node rebuild serialize on the same
    per-source lock, so concurrent calls for one source yield exactly one
    build. The pointer is refreshed either way: a rebuild also warms the next
    unfinalized ``finalize()`` on this node.
    """
    sdir = _schema_dir(_catalog_dir())
    skey = _source_key(header)
    pointer_path = sdir / ".bykey" / f"{skey}.fp"
    lock_path = sdir / ".bykey" / f"{skey}.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Cannot create catalog dir {sdir}: {exc}. Zephon needs a writable "
            "node-local directory for shard catalogs even with the cache "
            "disabled (resolution: {cache.root}/.catalog -> $ZEPHON_CATALOG_DIR "
            "-> {tempdir}/zephon-{uid}/catalog); set ZEPHON_CATALOG_DIR to "
            "override."
        ) from exc
    with FileLock(str(lock_path)):
        fp = expected_fp or catalog_io.read_pointer(pointer_path)
        if fp is None or not catalog_io.is_valid(
            sdir / fp, schema_version=SCHEMA_VERSION
        ):
            logger.info(
                "Building shard catalog for dataset %r (root=%s): first "
                "contact runs full discovery (listing + per-shard metadata "
                "reads, remote roots can take minutes); the result is cached "
                "under %s for later runs.",
                header.name,
                header.root,
                sdir,
            )
            build_start = time.perf_counter()
            built = build_catalog(header)
            logger.info(
                "Built shard catalog for dataset %r in %.1fs",
                header.name,
                time.perf_counter() - build_start,
            )
            if expected_fp is not None and built.fingerprint != expected_fp:
                raise CatalogFingerprintMismatch(
                    f"Rebuilt catalog fingerprint {built.fingerprint} != baked "
                    f"{expected_fp} for dataset {header.name!r}; discovery is "
                    "not deterministic across machines."
                )
            fp = built.fingerprint
            catalog_io.write_atomic(sdir / fp, built.file_bytes)
            del built  # drop the private build buffers; callers mmap the file
        if catalog_io.read_pointer(pointer_path) != fp:
            catalog_io.write_pointer(pointer_path, fp)
    return fp


def _load_and_register(fp: str) -> ShardCatalog:
    """Mmap the catalog file for ``fp`` and share it via the process registry.

    The mmap happens outside the registry lock; on a race the loser's mapping
    is dropped and the winner's is shared.
    """
    catalog = ShardCatalog(catalog_io.load_mmap(_schema_dir(_catalog_dir()) / fp))
    with _REGISTRY_LOCK:
        return _REGISTRY.setdefault((SCHEMA_VERSION, fp), catalog)


def finalize(handle: ShardCatalogHandle) -> str:
    """Build-or-load the catalog under the per-source lock; bake the fingerprint."""
    if handle.fingerprint is not None:
        attach(handle)
        return handle.fingerprint
    fp = _ensure_catalog_file(handle.dataset)
    _load_and_register(fp)
    handle.fingerprint = fp
    return fp


def attach(handle: ShardCatalogHandle) -> ShardCatalog:
    """Load the catalog for a finalized handle (registry -> file -> rebuild)."""
    if handle.fingerprint is None:
        raise RuntimeError(
            "attach() requires a finalized handle (fingerprint is the registry/"
            "file key). A missing fingerprint after unpickling means the "
            "Dataset was pickled before finalize() baked it — the Engine "
            "finalizes at construction; call finalize() first."
        )
    catalog = _REGISTRY.get((SCHEMA_VERSION, handle.fingerprint))
    if catalog is not None:
        return catalog
    path = _schema_dir(_catalog_dir()) / handle.fingerprint
    if not catalog_io.is_valid(path, schema_version=SCHEMA_VERSION):
        # Cross-node miss: rebuild from source, serialized with finalize().
        _ensure_catalog_file(handle.dataset, expected_fp=handle.fingerprint)
    return _load_and_register(handle.fingerprint)


def clear_registry() -> None:
    """Drop all in-process mmapped catalogs (tests/teardown)."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


__all__ = [
    "CatalogFingerprintMismatch",
    "DatasetHeader",
    "SCHEMA_VERSION",
    "ShardCatalogHandle",
    "attach",
    "clear_registry",
    "finalize",
    "resolve_catalog_dir",
    "set_catalog_dir",
]
