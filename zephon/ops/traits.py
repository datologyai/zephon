# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Operator traits that guide planning and scheduling."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class OpTraits:
    """Static capabilities an operator advertises to the planner.

    ``indexable`` — whether the operator preserves indexability through the
    plan.

    ``preserves_cursor_order`` — whether the operator preserves per-lane
    cursor order (no reordering across chunk/offset/lineage). Required —
    every operator author must make an explicit ``True`` (1:1 maps, payload
    transforms, non-reordering filters) or ``False`` (reorders, shuffles,
    packs) call.

    ``parallelism`` — suggested parallelism for the operator when not
    overridden.

    ``batch_shape_sensitive`` — if True, the operator's outputs can depend
    on how inputs are grouped into micro-batches (e.g., per-batch RNG or
    statistics). In deterministic mode, stages that contain at least one
    such operator will have time-based flush disabled to preserve strong
    determinism. When False, ordering determinism suffices and latency
    flush may be kept for performance.

    ``requires_serial_state`` — if True, the operator maintains
    cross-invocation state (e.g., buffers) that must be confined to a single
    operator instance for determinism. In deterministic mode, operators with
    this trait will automatically run with parallelism=1. This prevents
    nondeterministic behavior when multiple worker instances would each
    maintain separate buffers.
    """

    indexable: bool = True
    preserves_cursor_order: bool
    parallelism: int = 1
    batch_shape_sensitive: bool = False
    requires_serial_state: bool = False


__all__ = ["OpTraits"]
