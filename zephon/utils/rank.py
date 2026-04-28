# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Rank-aware context prefix for debug / stall / watchdog logs.

Reads ``RANK`` and ``LOCAL_RANK`` from the environment (set by torchrun,
torch.distributed.launch, etc.) at import time so every spawned worker
inherits the parent's values.  ``pid`` is read on each call because it
differs across spawned child processes.
"""

from __future__ import annotations

import os

_RANK: str = os.environ.get("RANK", "?")
_LOCAL_RANK: str = os.environ.get("LOCAL_RANK", "?")


def rank_ctx() -> str:
    """Return ``rank=X local=Y pid=Z`` for attribution in debug logs."""
    return f"rank={_RANK} local={_LOCAL_RANK} pid={os.getpid()}"


__all__ = ["rank_ctx"]
