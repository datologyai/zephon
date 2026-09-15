"""Small persisted-format fixtures independent of the installed LitData writer."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import optree

from zephon._internal.io.formats.litdata_support import support


def write_litdata_fixture(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    layout: str = "legacy",
    batch_sizes: tuple[int, ...] | None = None,
    ipc_compression: str | None = None,
) -> Path:
    """Write legacy int/string rows or a file/stream Arrow footer and index."""
    root.mkdir(parents=True, exist_ok=True)
    header_size = 4 * (len(rows) + 2)
    offsets = [header_size]
    body = b""
    if layout in {"legacy", "hybrid"}:
        for row in rows:
            text = row["text"].encode("utf-8")
            body += struct.pack("<IIq", 8, len(text), row["id"]) + text
            offsets.append(header_size + len(body))
    else:
        offsets *= len(rows) + 1
    data = struct.pack(f"<{len(rows) + 2}I", len(rows), *offsets) + body
    if layout != "legacy":
        import pyarrow as pa

        table = pa.Table.from_pylist(rows)
        sink = pa.BufferOutputStream()
        write = pa.ipc.new_stream if layout == "stream" else pa.ipc.new_file
        options = pa.ipc.IpcWriteOptions(compression=ipc_compression)
        with write(sink, table.schema, options=options) as writer:
            start = 0
            sizes = batch_sizes
            if sizes is None:
                sizes = tuple(min(256, len(rows) - i) for i in range(0, len(rows), 256))
            for size in sizes:
                batch = pa.RecordBatch.from_arrays(
                    [
                        column.combine_chunks().slice(start, size)
                        for column in table.columns
                    ],
                    schema=table.schema,
                )
                writer.write_batch(batch)
                start += size
            assert start == len(rows)
        ipc = sink.getvalue().to_pybytes()
        data += ipc + struct.pack("<I", len(ipc)) + b"LDARW01\0"
    path = root / "chunk-0-0.bin"
    path.write_bytes(data)
    config = {
        "chunk_size": len(rows),
        "chunk_bytes": None,
        "compression": None,
        "data_format": ["int", "str"],
        "data_spec": support.treespec_dumps(optree.tree_structure(rows[0])),
        "item_loader": "PyTreeLoader",
        "encryption": None,
    }
    (root / "index.json").write_text(
        json.dumps(
            {
                "config": config,
                "chunks": [
                    {
                        "filename": path.name,
                        "chunk_size": len(rows),
                        "chunk_bytes": len(data),
                        "dim": None,
                    }
                ],
            }
        )
    )
    return path
