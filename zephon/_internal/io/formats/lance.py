# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Lance dataset discovery and indexed sample reads.

Zephon reads Lance through its open formats rather than its dataset library.
The table manifest (parsed in :mod:`lance_support.manifest`, no ``pylance``) is the
shard index; one fragment is one shard, and that fragment's ``.lance`` data
file flows through the ordinary resolver/cache path like a Vortex or Parquet
shard. ``pylance`` is used only to decode pages, through its file-level reader.

A fragment is the only sensible shard: it is the dataset's unit of horizontal
slicing, with its own row count and its own file. It cannot be split (two
shards would share one cached file) nor merged (two fragments are two files).
What a fragment may *not* do here is spread its columns over several data
files, which ``add_columns`` produces: a shard resolves exactly one file, so
such a fragment is refused at discovery with the remedy in the message.
Deleted rows are supported — the shard reports the surviving row count and maps
each logical row onto its physical offset through the fragment's deletion
vector.

Unsupported cases are refused at discovery rather than mid-run.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Iterator, Mapping

import numpy as np

from zephon._internal.io.formats.arrow_rows import (
    arrow_table_to_samples,
    configure_pyarrow,
)
from zephon._internal.io.formats.base import FormatHandler, ShardOpener, register_format
from zephon._internal.io.formats.lance_support.deletions import (
    DeletionVector,
    read_deletion_vector,
)
from zephon._internal.io.formats.lance_support.manifest import (
    DATA_DIR,
    DELETIONS_DIR,
    FLAG_DATA_OVERLAYS,
    FLAG_FRAGMENT_TREE,
    KNOWN_READER_FLAGS,
    LanceFragment,
    LanceManifest,
    dataset_root_of,
    is_manifest_path,
    latest_manifest_path,
    read_manifest,
)
from zephon._internal.io.formats.metadata_cache import MetadataCache, file_cache_key
from zephon._internal.io.protocols import RandomAccessShard
from zephon._internal.io.storage import RouterStorageBackend, StorageBackend
from zephon._internal.io.types import LocalShardRef, ShardFile, ShardLocator
from zephon.types import SamplePayload

if TYPE_CHECKING:
    from zephon._internal.io.catalog import CatalogSet
    from zephon.io.dataset import Dataset
    from zephon.io.options import StoreOptions

try:
    import lance.file as _lance_file
except ImportError:
    _lance_file = None


def _require_lance() -> ModuleType:
    """Return pylance's file-level reader module or explain how to install it."""
    if _lance_file is None:
        raise ImportError(
            "Reading Lance data files requires the 'pylance' package; "
            + 'install it with: pip install -U "zephon[lance]"'
        )
    return _lance_file


# ``compact_files()`` skips a fragment it cannot merge, so only a rewrite always works.
_REWRITE = (
    "rewrite the dataset (e.g. lance.write_dataset(lance.dataset(src), dst)) "
    "so each fragment is one file holding exactly the schema's columns"
)


@contextmanager
def _local_io_errors(path: Path) -> Iterator[None]:
    """Surface a missing or inaccessible shard file as the path's ``OSError``.

    pylance reports a missing file as ``ValueError``, or in older releases as a
    bare ``OSError``; the eviction retry around shard reads catches only
    ``OSError``, and should re-resolve only when the path itself failed.
    """
    try:
        yield
    except (ValueError, RuntimeError, OSError) as exc:
        try:
            path.stat()
        except OSError as io_error:
            raise io_error from exc
        raise


def _reject(manifest_path: str, reason: str) -> ValueError:
    return ValueError(f"Unsupported Lance dataset at {manifest_path}: {reason}")


@dataclass(frozen=True)
class _FragmentPlan:
    """Everything one shard needs to read a fragment, derived from the manifest."""

    basename: str
    file_bytes: int
    physical_rows: int
    num_rows: int
    # Dataset column name -> its column number in the file. The name is the
    # manifest's, which a rename updates without rewriting the file.
    columns: tuple[tuple[str, int], ...]
    file_columns: int
    deletion_basename: str | None
    num_deleted_rows: int


def _validate_schema(manifest: LanceManifest, manifest_path: str) -> None:
    if not manifest.fields:
        raise _reject(manifest_path, "the manifest has an empty schema")
    unknown = manifest.reader_feature_flags & ~KNOWN_READER_FLAGS
    if unknown:
        raise _reject(
            manifest_path,
            f"it sets reader feature flags {unknown:#x} that this version of "
            "Zephon does not understand; upgrade Zephon to read it",
        )
    if manifest.has_fragment_tree or manifest.reader_feature_flags & FLAG_FRAGMENT_TREE:
        raise _reject(
            manifest_path,
            "its fragments live in an external fragment tree rather than in "
            "the manifest, which Zephon cannot walk yet",
        )
    if manifest.reader_feature_flags & FLAG_DATA_OVERLAYS:
        raise _reject(
            manifest_path,
            "it carries data overlay files; reading the base values alone would "
            "return rows that a later update has already replaced",
        )
    for field in manifest.fields:
        if field.is_blob:
            raise _reject(
                manifest_path,
                f"column {field.name!r} is a Lance blob column, whose values "
                "live outside the fragment data files",
            )


def _fragment_plan(
    manifest: LanceManifest, fragment: LanceFragment, manifest_path: str
) -> _FragmentPlan:
    """Validate one fragment and resolve its file, columns and tombstones."""
    where = f"fragment {fragment.id}"
    if fragment.has_overlays:
        raise _reject(
            manifest_path,
            f"{where} carries data overlay files; reading the base values alone "
            "would return stale rows",
        )
    if fragment.physical_rows <= 0:
        raise _reject(manifest_path, f"{where} has no recorded row count")

    # A file holding only system columns (row lineage spilled out of the
    # fragment metadata) stores no dataset column, so it is simply not read.
    data_files = [f for f in fragment.files if f.holds_dataset_columns()]
    if len(data_files) != 1:
        detail = (
            f"spans {len(data_files)} data files (e.g. after add_columns)"
            if data_files
            else "has no data file holding dataset columns"
        )
        raise _reject(
            manifest_path,
            f"{where} {detail}; a Zephon shard resolves one file, so {_REWRITE}",
        )
    data_file = data_files[0]
    if data_file.has_base_id:
        raise _reject(
            manifest_path,
            f"{where}'s data file lives under another dataset's base path",
        )
    if data_file.is_legacy_v1:
        raise _reject(
            manifest_path,
            f"{where} uses the legacy v1 Lance file format; rewrite it with "
            "data_storage_version 2.0 or newer",
        )
    if "://" in data_file.path or os.path.isabs(data_file.path):
        raise _reject(manifest_path, f"{where}'s data file lives outside the dataset")

    order = manifest.column_order(data_file.field_ids)
    if None in order:
        # The file keeps a dropped column, but the schema no longer says its width.
        raise _reject(
            manifest_path,
            f"{where} still stores a dropped column; {_REWRITE}",
        )
    columns: list[tuple[str, int]] = []
    # fmt: off
    for field in manifest.top_level_fields():
        if field.id not in order:
            raise _reject(
                manifest_path,
                f"{where} does not store column {field.name!r}",
            )
        columns.append((field.name, order.index(field.id)))
    # fmt: on

    deletion = fragment.deletion
    if deletion is not None and deletion.has_base_id:
        raise _reject(
            manifest_path,
            f"{where}'s deletion file lives under another dataset's base path",
        )
    return _FragmentPlan(
        basename=data_file.path,
        file_bytes=data_file.file_size_bytes,
        physical_rows=fragment.physical_rows,
        num_rows=fragment.num_rows,
        columns=tuple(columns),
        file_columns=len(order),
        deletion_basename=deletion.basename if deletion else None,
        num_deleted_rows=deletion.num_deleted_rows if deletion else 0,
    )


@dataclass(frozen=True)
class _DatasetPlan:
    """One pinned manifest resolved into per-fragment read plans."""

    root: str
    fragments: tuple[_FragmentPlan, ...]


def _plan_dataset(storage: StorageBackend, manifest_path: str) -> _DatasetPlan:
    """Read a pinned manifest and validate every fragment it lists."""
    manifest = read_manifest(storage, manifest_path)
    _validate_schema(manifest, manifest_path)
    if not manifest.fragments:
        raise _reject(manifest_path, "the dataset has no fragments")
    return _DatasetPlan(
        root=dataset_root_of(manifest_path),
        fragments=tuple(
            _fragment_plan(manifest, fragment, manifest_path)
            for fragment in manifest.fragments
        ),
    )


@dataclass(frozen=True)
class _FragmentMetadata:
    """What a fragment's reader needs beyond the manifest, decoded once.

    Holds no file handle: a retained reader would keep an evicted shard's
    disk space allocated behind the shard cache's back, and pylance's reader
    serializes concurrent takes.
    """

    # File column names to project to, or None when the file holds exactly
    # the dataset's columns in order.
    projection: list[str] | None
    # Set when the file's own column names differ from the dataset's, which a
    # rename produces: Lance updates the manifest and leaves the file alone.
    rename: tuple[str, ...] | None
    deletion: DeletionVector | None


def _fragment_metadata(
    probe: Any, path: Path, plan: _FragmentPlan, deletion: DeletionVector | None
) -> _FragmentMetadata:
    """Map the dataset's columns onto ``probe``'s file columns by position."""
    file_names: list[str] = list(probe.metadata().schema.names)
    if len(file_names) != plan.file_columns:
        raise ValueError(
            f"Lance file {path} holds {len(file_names)} columns, but its "
            f"manifest entry accounts for {plan.file_columns}"
        )
    # Lance rejects duplicate field names at write time, so projecting the
    # positions back to names is unambiguous.
    wanted = [file_names[column] for _name, column in plan.columns]
    names = tuple(name for name, _column in plan.columns)
    return _FragmentMetadata(
        projection=None if wanted == file_names else wanted,
        rename=None if tuple(wanted) == names else names,
        deletion=deletion,
    )


class LanceShard(RandomAccessShard):
    """Random-access shard backed by one fragment's ``.lance`` data file."""

    def __init__(
        self, reader: Any, metadata: _FragmentMetadata, *, length: int
    ) -> None:
        self._reader: Any | None = reader
        self._metadata = metadata
        self._length = length

    def __getitem__(self, index: int) -> SamplePayload:
        return self.getsamples([index])[0]

    def __len__(self) -> int:
        return self._length

    def close(self) -> None:
        # pylance's reader has no close(); dropping the last reference
        # releases its file handle.
        self._reader = None

    def getsamples(self, indices: list[int]) -> list[SamplePayload]:
        """Bulk read rows in request order, preserving duplicate indices."""
        if not indices:
            return []
        for i in indices:
            if i < 0 or i >= self._length:
                raise IndexError(i)
        reader = self._reader
        if reader is None:
            raise RuntimeError("LanceShard has been closed")

        metadata = self._metadata
        deletion = metadata.deletion
        rows = deletion.to_physical(indices) if deletion else indices
        # The file reader requires strictly increasing indices, so duplicate and
        # out-of-order requests read each row once and are reordered afterwards.
        if all(a < b for a, b in zip(rows, rows[1:])):
            wanted = rows
            positions: Any = range(len(rows))
        else:
            wanted = sorted(set(rows))
            position_of = {row: pos for pos, row in enumerate(wanted)}
            positions = map(position_of.__getitem__, rows)

        # Checked on physical rows: logical rows that are contiguous straddle any
        # tombstone between them.
        if wanted[-1] - wanted[0] + 1 == len(wanted):
            table = reader.read_range(wanted[0], len(wanted)).to_table()
        else:
            table = reader.take_rows(wanted).to_table()
        if metadata.rename is not None:
            table = table.rename_columns(list(metadata.rename))
        return arrow_table_to_samples(table, positions)


class LanceShardOpener:
    """Share manifests and decoded fragment metadata across a store's shards."""

    def __init__(self, options: "StoreOptions", storage: StorageBackend) -> None:
        _require_lance()
        configure_pyarrow(options.arrow)
        # Manifests and tombstones live beside the data rather than in the
        # shard cache, so they are read through the store's own backend.
        self._storage = storage
        # One entry per Lance dataset in the store, and the only thing that must
        # survive ``metadata_cache_entries=0``: without it every read would
        # refetch the manifest.
        self._plans: dict[str, _DatasetPlan] = {}
        self._plans_lock = threading.Lock()
        self._metadata = MetadataCache[str, _FragmentMetadata](
            options.metadata_cache_entries
        )

    def _plan(self, manifest_path: str) -> _DatasetPlan:
        with self._plans_lock:
            plan = self._plans.get(manifest_path)
            if plan is None:
                plan = _plan_dataset(self._storage, manifest_path)
                self._plans[manifest_path] = plan
            return plan

    def _read_deletions(
        self, root: str, fragment: _FragmentPlan
    ) -> DeletionVector | None:
        # The tombstone file is small and lives beside the data under the
        # dataset root, not in the shard cache, so it is read through storage.
        if fragment.deletion_basename is None:
            return None
        return read_deletion_vector(
            self._storage,
            os.path.join(root, DELETIONS_DIR, fragment.deletion_basename),
            expected_deleted=fragment.num_deleted_rows,
            physical_rows=fragment.physical_rows,
        )

    def _open_reader(
        self, path: Path, root: str, fragment: _FragmentPlan, cache_key: str
    ) -> tuple[Any, _FragmentMetadata]:
        lance_file = _require_lance()
        probe = None

        def load_metadata() -> _FragmentMetadata:
            nonlocal probe
            probe = lance_file.LanceFileReader(str(path))
            return _fragment_metadata(
                probe, path, fragment, self._read_deletions(root, fragment)
            )

        with _local_io_errors(path):
            metadata = self._metadata.get_or_load(cache_key, load_metadata)
            # Reuse the reader opened on a metadata miss when it needs no projection.
            if probe is not None and metadata.projection is None:
                return probe, metadata
            reader = lance_file.LanceFileReader(str(path), columns=metadata.projection)
        return reader, metadata

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard:
        """Open a shard against this store's manifest and fragment caches."""
        extra = local_ref.extra or {}
        manifest_path = extra.get("manifest")
        if not isinstance(manifest_path, str):
            raise ValueError(
                f"Lance shard {locator.shard_id} is missing its manifest path"
            )
        plan = self._plan(manifest_path)
        # Discovery numbers shards by their position in the manifest's fragment
        # list, so the shard id indexes it directly.
        fragment = plan.fragments[int(locator.shard_id)]
        path = local_ref.raw.path
        # Key on the version too: two datasets pinned to different versions of
        # one directory share their unchanged data files, and without the cache
        # they share the local path as well, but a delete or a rename between
        # them changes which rows or names that file yields.
        reader, metadata = self._open_reader(
            path, plan.root, fragment, f"{manifest_path}\0{file_cache_key(path)}"
        )
        return LanceShard(reader, metadata, length=fragment.num_rows)

    def close(self) -> None:
        """Release this store's fragment metadata and manifests."""
        self._metadata.clear()
        with self._plans_lock:
            self._plans.clear()


class LanceFormat(FormatHandler):
    """Format handler for Lance datasets (one shard per fragment)."""

    kind = "lance"

    @contextmanager
    def create_opener(
        self,
        catalog_set: "CatalogSet | None",
        options: "StoreOptions",
        storage: StorageBackend,
    ) -> Iterator[ShardOpener]:
        """Provide a store-scoped opener and release its readers on exit."""
        opener = LanceShardOpener(options, storage)
        try:
            yield opener
        finally:
            opener.close()

    def resolve_root(self, path: str, storage: StorageBackend) -> str:
        """Pin ``path`` to one dataset version by resolving its latest manifest.

        Manifests are immutable, so the pinned root keys the catalog cache and
        every node's rebuild to exactly one version. A manifest path is already
        pinned and is returned unchanged.
        """
        if is_manifest_path(path):
            return path
        return latest_manifest_path(storage, path)

    def discover(
        self, path: str, storage: StorageBackend
    ) -> tuple[Mapping[int, int], Mapping[int, Mapping[str, Any]]]:
        """Read the manifest and return one shard per fragment, in manifest order.

        Needs neither ``pylance`` nor a single data file: the manifest records
        each fragment's data file, size and row count.
        """
        manifest_path = self.resolve_root(path, storage)
        plan = _plan_dataset(storage, manifest_path)
        data_root = os.path.join(plan.root, DATA_DIR)

        shard_index: dict[int, int] = {}
        shard_meta: dict[int, dict[str, Any]] = {}
        for shard_id, fragment in enumerate(plan.fragments):
            size = fragment.file_bytes
            if size <= 0:
                # Older writers left file_size_bytes unset.
                size = int(
                    storage.stat(os.path.join(data_root, fragment.basename))["size"]
                )
            shard_index[shard_id] = fragment.num_rows
            shard_meta[shard_id] = {
                "raw": {"basename": fragment.basename, "bytes": size, "hashes": {}},
                # Identical for every shard, so the catalog stores it once; the
                # opener rereads the pinned manifest for the rest.
                "extra": {"manifest": manifest_path},
            }
        return shard_index, shard_meta

    def discover_counts(
        self, path: str, storage: StorageBackend
    ) -> tuple[np.ndarray, np.ndarray]:
        """Count-only discovery: read each fragment's row count from the manifest.

        Skips the per-shard locator metadata and the data-file ``stat`` that
        ``discover`` falls back to when a writer left a file size unset. Runs
        the same validation, so it refuses what ``discover`` refuses.
        """
        plan = _plan_dataset(storage, self.resolve_root(path, storage))
        counts = [fragment.num_rows for fragment in plan.fragments]
        return np.arange(len(counts), dtype=np.int64), np.array(counts, dtype=np.int64)

    def build_locators(self, dataset: "Dataset") -> Mapping[int, ShardLocator]:
        """Build locators rooted at the pinned version's ``data/`` directory."""
        path = dataset.backend.get("path")
        if not isinstance(path, str):
            raise ValueError("Lance dataset missing 'path' in backend metadata")
        shards = dataset.backend.get("shards")
        if not isinstance(shards, Mapping):
            raise ValueError("Lance dataset missing 'shards' metadata")
        data_root = os.path.join(dataset_root_of(path), DATA_DIR)

        locators: dict[int, ShardLocator] = {}
        for shard_id_obj, shard_meta in shards.items():
            shard_id = int(shard_id_obj)
            if not isinstance(shard_meta, Mapping):
                raise ValueError(f"Invalid shard metadata for shard {shard_id}")
            raw_meta = shard_meta.get("raw")
            if not isinstance(raw_meta, Mapping):
                raise ValueError(f"Shard {shard_id} missing Lance raw metadata")
            extra = shard_meta.get("extra")
            locators[shard_id] = ShardLocator(
                dataset=dataset.name,
                shard_id=shard_id,
                format=self.kind,
                root=data_root,
                raw=ShardFile(
                    basename=str(raw_meta["basename"]),
                    bytes=int(raw_meta["bytes"]),
                    hashes={},
                ),
                zip=None,
                compression=None,
                extra=extra if isinstance(extra, Mapping) else None,
            )
        return locators

    def open_shard(
        self, locator: ShardLocator, local_ref: LocalShardRef
    ) -> RandomAccessShard:
        """Open one shard without a store (stores go through ``create_opener``).

        Builds a throwaway opener, so the manifest is reread per call.
        """
        from zephon.io.options import StoreOptions

        opener = LanceShardOpener(StoreOptions(), RouterStorageBackend())
        return opener.open_shard(locator, local_ref)


register_format(LanceFormat())

__all__ = ["LanceFormat", "LanceShard", "LanceShardOpener"]
