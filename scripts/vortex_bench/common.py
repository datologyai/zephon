# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Untimed provenance and correctness helpers shared by the benchmark tools."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Set these before importing native libraries, including in conversion processes.
for name in (
    "VORTEX_MAX_THREADS",
    "ZEPHON_VORTEX_THREADS",
    "ZEPHON_PARQUET_ARROW_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "ARROW_NUM_THREADS",
):
    os.environ[name] = "1"

SCRIPT_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("VORTEX_BENCH_ROOT", SCRIPT_ROOT)).expanduser().resolve()


def file_digest(path: Path) -> str:
    """Hash a file without keeping its contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_digest(row: dict[str, Any], columns: dict[str, str]) -> str:
    """Compare list-valued main output and NumPy-valued candidate output fairly."""
    import numpy as np

    digest = hashlib.sha256()
    for name, kind in columns.items():
        digest.update(name.encode())
        value = row[name]
        if kind.startswith("array:"):
            data = np.asarray(value, dtype=kind.split(":", 1)[1]).tobytes()
        elif kind == "str":
            data = value.encode("utf-8")
        else:
            data = json.dumps(int(value)).encode()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically publish a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
