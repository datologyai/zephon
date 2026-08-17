# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Read and write decoded row-group cache files using the Arrow file format."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zephon._internal.io.formats.arrow_rows import require_pyarrow, take_and_materialize

_IDENTITY_METADATA_KEY = b"zephon.decoded_rg.identity"
_VERSION_METADATA_KEY = b"zephon.decoded_rg.payload_version"
_PAYLOAD_VERSION = b"1"


@dataclass(frozen=True)
class DecodedRGIdentity:
    """Fingerprint tying a cached row group to its source and catalog.

    ``ParquetRGIndex`` creates it, and ``ArrowRGFileCodec`` stores and validates it.
    """

    digest: str

    def __post_init__(self) -> None:
        if not self.digest.startswith("sha256:") or len(self.digest) != 71:
            raise ValueError("Decoded RG identity must be a sha256 digest")
        try:
            bytes.fromhex(self.digest[7:])
        except ValueError as exc:
            raise ValueError("Decoded RG identity must be a sha256 digest") from exc


class ArrowRGHandle:
    """One opened Arrow row-group cache file.

    The handle keeps its mmap, reader, and Arrow table alive until requested rows
    are materialized, then closes those resources together.
    """

    def __init__(self, source: Any, reader: Any, table: Any) -> None:
        self._source: Any | None = source
        self._reader: Any | None = reader
        self._table: Any | None = table
        self._closed = False

    def take_and_materialize(
        self,
        local_indices: list[int],
    ) -> list[dict[str, object]]:
        if self._closed:
            raise RuntimeError("Decoded RG payload mapping is closed")
        return take_and_materialize(self._table, local_indices)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._table = None
        self._reader = None
        source = self._source
        self._source = None
        if source is not None:
            with contextlib.suppress(Exception):
                source.close()

    def __enter__(self) -> "ArrowRGHandle":
        if self._closed:
            raise RuntimeError("Decoded RG payload mapping is closed")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort at shutdown
        with contextlib.suppress(BaseException):
            self.close()


class ArrowRGFileCodec:
    """File-format helper for cached decoded Arrow row groups.

    ``ParquetRGCache`` uses it to measure and write whole Arrow tables and to open
    validated ``ArrowRGHandle`` objects; it does not decode Parquet or build rows.
    """

    format_version = 1
    suffix = ".arrow"

    def measure(self, table: Any, identity: DecodedRGIdentity) -> int:
        """Measure exact writer output without allocating a payload buffer."""
        pa = require_pyarrow()
        # Admission reserves exact capacity before allowing the temporary write.
        measured = pa.MockOutputStream()
        table_with_identity = self._table_with_identity(table, identity)
        with pa.ipc.new_file(
            measured,
            table_with_identity.schema,
            options=self._writer_options(pa),
        ) as writer:
            writer.write_table(table_with_identity)
        return measured.size()

    def write(
        self,
        table: Any,
        identity: DecodedRGIdentity,
        destination: str | os.PathLike[str],
        *,
        exact_bytes: int,
    ) -> int:
        """Stream directly to a private temp path and verify exact size."""
        if exact_bytes <= 0:
            raise ValueError("Decoded RG exact payload bytes must be positive")
        pa = require_pyarrow()
        path = Path(destination)
        table_with_identity = self._table_with_identity(table, identity)
        sink = path.open("xb")
        try:
            with sink:
                with pa.ipc.new_file(
                    sink,
                    table_with_identity.schema,
                    options=self._writer_options(pa),
                ) as writer:
                    writer.write_table(table_with_identity)
                actual_bytes = sink.tell()
            if actual_bytes != exact_bytes:
                raise RuntimeError(
                    "Decoded RG payload measurement mismatch: "
                    + f"expected {exact_bytes}, wrote {actual_bytes}"
                )
            return actual_bytes
        except BaseException:
            with contextlib.suppress(OSError):
                path.unlink()
            raise

    def open_mapped(
        self,
        source: str | os.PathLike[str],
        identity: DecodedRGIdentity,
        *,
        exact_bytes: int,
    ) -> ArrowRGHandle:
        """Map and validate one immutable cache file, returning its open handle."""
        pa = require_pyarrow()
        path = Path(source)
        mapped = pa.memory_map(os.fspath(path), "r")
        try:
            if mapped.size() != exact_bytes:
                raise RuntimeError(f"Decoded RG payload size mismatch at {path}")
            reader = pa.ipc.open_file(mapped)
            metadata = reader.schema.metadata or {}
            if metadata.get(_VERSION_METADATA_KEY) != _PAYLOAD_VERSION:
                raise RuntimeError(f"Decoded RG payload version mismatch at {path}")
            if metadata.get(_IDENTITY_METADATA_KEY) != identity.digest.encode("ascii"):
                raise RuntimeError(f"Decoded RG payload identity mismatch at {path}")
            table = reader.read_all()
            return ArrowRGHandle(mapped, reader, table)
        except BaseException:
            with contextlib.suppress(Exception):
                mapped.close()
            raise

    @staticmethod
    def _writer_options(pa: Any) -> Any:
        return pa.ipc.IpcWriteOptions(compression=None)

    @staticmethod
    def _table_with_identity(table: Any, identity: DecodedRGIdentity) -> Any:
        metadata = dict(table.schema.metadata or {})
        metadata[_VERSION_METADATA_KEY] = _PAYLOAD_VERSION
        metadata[_IDENTITY_METADATA_KEY] = identity.digest.encode("ascii")
        return table.replace_schema_metadata(metadata)


__all__ = ["ArrowRGFileCodec", "ArrowRGHandle", "DecodedRGIdentity"]
