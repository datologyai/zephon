# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Routing metadata and plan-time rules for ``Pipeline.branch()``.

A ``branch()`` call adds a ``Split`` vertex after the op before the branches
(U) and a ``Merge`` vertex after the branches. The split holds a
:class:`Router`. Each branch op is an ordinary graph node whose only input is
the split; a passthrough branch gets a built-in identity op. The op added after
the branches (J) takes the merge as its only input.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from zephon.ops.accumulators import CountingAccumulator, PassthroughAccumulator
from zephon.types import SamplePayload

if TYPE_CHECKING:
    from zephon._internal.graph import Node, Stage

# Runners that can execute a stage holding branch ops. Each runner joins this
# set in the change that wires routing and the BranchAggregator into it.
#
# Empty for now: so far only the graph, builder and planner know about
# branches. No runner routes U's output or merges branch results yet. A runner
# without that support would wire a branched stage as a chain and feed every
# record through every branch op, so require_runner_support() refuses it
# instead.
# TODO(branching): add "inline" once the inline runner routes and joins
# branches, then "threads", "process" and "remote" as each one is wired.
SUPPORTED_RUNNERS: frozenset[str] = frozenset()

_BATCHING_ONLY_ACCUMULATORS = (CountingAccumulator, PassthroughAccumulator)


@dataclass(frozen=True)
class Router:
    """Routes each record a split receives to exactly one named branch.

    Attributes:
        by: Returns a branch name for a record's payload. It must be
            deterministic and read only the payload.
        branches: Branch names in declaration order.
        default: Branch for keys that name no branch; ``None`` raises.
    """

    by: Callable[[SamplePayload], str]
    branches: tuple[str, ...]
    default: str | None = None
    _names: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate names and cache them for lookups."""
        if not self.branches:
            raise ValueError("a router needs at least one branch")
        names = frozenset(self.branches)
        if len(names) != len(self.branches):
            raise ValueError(f"branch names must be unique: {list(self.branches)}")
        if self.default is not None and self.default not in names:
            raise ValueError(
                f"default={self.default!r} names no branch; "
                f"branches: {list(self.branches)}"
            )
        object.__setattr__(self, "_names", names)

    def route(self, payload: SamplePayload) -> str:
        """Return the branch name for ``payload``."""
        key = self.by(payload)
        if key in self._names:
            return key
        if self.default is not None:
            return self.default
        raise ValueError(
            f"branch router returned {key!r}, which names no branch "
            f"(branches: {list(self.branches)}); return one of them or pass "
            "default= to branch()"
        )

    def identity(self) -> tuple[Any, ...]:
        """Return a stable, hashable description for plan fingerprints."""
        by = self.by
        return (
            getattr(by, "__module__", None),
            getattr(by, "__qualname__", type(by).__qualname__),
            self.branches,
            self.default,
        )


def stateless_violation(node: "Node[Any]") -> str | None:
    """Return why ``node`` cannot run inside a branch, or ``None`` if it can.

    A branch op must give the same output however the runner cuts its input
    into batches, because the runner flushes it after every slice: every
    output descends from one input in input order, its traits do not depend
    on batch shape or serial state, and its accumulator only sets batch
    boundaries.
    """
    traits = node.op.traits()
    if not traits.preserves_cursor_order:
        return "it does not preserve cursor order (preserves_cursor_order=False)"
    if traits.batch_shape_sensitive:
        return "its output depends on batch shape (batch_shape_sensitive=True)"
    if traits.requires_serial_state:
        return "it keeps serial state across calls (requires_serial_state=True)"
    try:
        accumulator = node.op.accumulator(deterministic=True, ctx={})
    except Exception as exc:  # noqa: BLE001 - report any construction failure
        return f"its accumulator could not be built at plan time ({exc!r})"
    if type(accumulator) not in _BATCHING_ONLY_ACCUMULATORS:
        return (
            f"its accumulator {type(accumulator).__name__} may hold records "
            "across batches; branch ops need a CountingAccumulator or "
            "PassthroughAccumulator"
        )
    return None


def require_runner_support(stage: "Stage", runner_type: str) -> None:
    """Raise if ``stage`` holds a split or merge that ``runner_type`` cannot run.

    The planner places each split and merge in the stage that holds their
    branch ops, so only that stage routes and rejoins. A stage that ends with
    U hands its output on as usual.
    """
    if runner_type in SUPPORTED_RUNNERS:
        return
    if not (stage.splits or stage.merges):
        return
    supported = ", ".join(sorted(SUPPORTED_RUNNERS)) or "none yet"
    raise NotImplementedError(
        f"stage {stage.name!r} uses Pipeline.branch(), which the "
        f"{runner_type!r} runner cannot run yet (supported runners: {supported})"
    )
