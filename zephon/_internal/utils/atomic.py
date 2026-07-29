# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Atomic single-file writes via a temp sibling and ``os.replace``."""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path


def atomic_write_bytes(
    path: str | Path,
    data: bytes,
    *,
    fsync: bool = False,
    unique_tmp: bool = True,
) -> None:
    """Atomically write ``data`` to ``path`` (temp sibling + ``os.replace``).

    Args:
        path: Destination file; parent directories are created if missing.
        data: Bytes to write.
        fsync: ``fsync`` the temp file before the rename, so a crash cannot
            leave a renamed-but-empty file. The parent directory is never
            ``fsync``ed.
        unique_tmp: Suffix the temp file with the writer's PID and thread id so
            concurrent writers to the same ``path`` (in different processes or
            different threads of one process) cannot clobber each other's temp
            file. Pass ``False`` when writes to ``path`` are externally
            serialized (e.g. under a per-shard lock) and a stable temp name is
            preferred.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".{os.getpid()}.{threading.get_ident()}.tmp" if unique_tmp else ".tmp"
    tmp = path.parent / f"{path.name}{suffix}"
    try:
        with open(tmp, "wb") as fobj:
            fobj.write(data)
            if fsync:
                fobj.flush()
                os.fsync(fobj.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
