# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Operator traits that guide planning and scheduling."""

from dataclasses import dataclass


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
        requires_serial_state: If True, the operator maintains cross-invocation
            state (e.g., buffers) that must be confined to a single operator
            instance for determinism. In deterministic mode, operators with this
            trait will automatically run with parallelism=1. This prevents
            nondeterministic behavior when multiple worker instances would each
            maintain separate buffers.
        stall_on_epoch_boundary: If True, the runner preserves the accumulator's
            buffer across epoch boundaries (flush sentinels) instead of calling
            ``flush(reset=True)`` immediately. The sentinel is stalled
            behind buffered data until ``try_epoch_reset()`` can release it.
            In the current design this is intentionally narrow: built-in support
            is limited to ``Batch(drop_last=True)``. General stalled operators
            would need extra replay-capsule state to make checkpoint/restore
            exact after eviction.
    """

    indexable: bool = True
    preserves_cursor_order: bool | None = None
    parallelism: int = 1
    batch_shape_sensitive: bool = False
    requires_serial_state: bool = False
    stall_on_epoch_boundary: bool = False
