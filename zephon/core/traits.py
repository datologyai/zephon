# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Operator traits that guide planning, scheduling, and buffering."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OpTraits:
    """Static capabilities an operator advertises to the planner.

    Attributes:
        indexable: Whether the operator preserves indexability through the plan.
        preserves_cursor_order: Whether the operator preserves per-lane cursor
            order (no reordering across chunk/offset/lineage). Required for
            deciding when the cursor-order notify path is safe.
        parallelism: Suggested parallelism for the operator when not overridden.
        batch_shape_sensitive: If True, the operator's outputs can depend on how
            inputs are grouped into micro-batches (e.g., per-batch RNG or
            statistics). In deterministic mode, stages that contain at least one
            such operator will have time-based flush disabled to preserve strong
            determinism. When False, ordering determinism suffices and latency
            flush may be kept for performance.
    """

    indexable: bool = True
    preserves_cursor_order: bool | None = None
    parallelism: int = 1
    batch_shape_sensitive: bool = False


@dataclass
class Buffering:
    """Runtime buffering preferences exposed by the operator implementation."""

    max_batch: int = 32
    max_latency_ms: Optional[int] = 5
