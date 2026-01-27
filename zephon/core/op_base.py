# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Abstract operator contracts shared by the planner and runtime."""

from typing import Any, Protocol, TypeVar

from zephon.core.accumulators import Accumulator
from zephon.core.constants import StreamItem
from zephon.core.traits import OpTraits


class OpContext:
    """Container exposing runner-provided services to operator instances."""

    def __init__(self, services: dict[str, Any]):
        self._services = services

    def get(self, key: str, default: Any | None = None) -> Any:
        """Fetch a service by name, returning ``default`` when unavailable."""
        return self._services.get(key, default)


InT = TypeVar("InT")
OutT = TypeVar("OutT", bound=StreamItem)


class Op(Protocol[InT, OutT]):
    """Protocol that Zephon operators must satisfy to plug into the pipeline.

    Fan-out operators **must** keep ``SampleMeta.sample_id`` stable and derive new
    lineage paths via :meth:`zephon.core.constants.SampleMeta.child`. This ensures
    that ordering and replay checks observe a deterministic total order even when
    operators execute with parallel workers.

    Accumulators and Deterministic Parallelism
    ------------------------------------------
    Each operator provides an accumulator via the ``accumulator()`` method. The
    accumulator runs on the pump thread (serial) and defines invocation boundaries
    for parallel workers. This ensures deterministic execution:

    - All cross-invocation state lives in the accumulator
    - Worker ``process_many`` calls are stateless across invocations
    - Thread and Process runners execute identical batch boundaries
    """

    def setup(
        self,
        ctx: OpContext,
        stage_index: int,
        stage_name: str,
        op_index: int,
        collect_stats: bool,
    ) -> None: ...

    def traits(self) -> OpTraits: ...

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[InT]:
        """Return the accumulator for this operator.

        The accumulator runs on the pump thread and defines invocation batch
        boundaries. For stateless operators, use PassthroughAccumulator.
        For operators that need buffering, use CountingAccumulator or a
        custom accumulator.

        Args:
            deterministic: If True, the accumulator should disable any
                non-deterministic behavior (e.g., time-based flushing).
            ctx: Context dictionary containing runtime services (e.g., mixture
                weights, dataset mappings, component ID lookups).

        Returns:
            An accumulator instance that will be used by the runner.
        """
        ...

    # Concrete ops specify precise input/output element types.
    def process_one(self, elem: InT) -> list[OutT]: ...

    def process_many(self, elems: list[InT]) -> list[OutT]: ...


class DefaultSetup:
    """Mixin providing a ``setup`` implementation that creates basic handles to stage metadata."""

    def __init__(self):
        self.stage_index = -1
        self.stage_name = ""
        self.op_index = -1
        self.collect_stats = False

    def setup(
        self,
        ctx: OpContext,
        stage_index: int,
        stage_name: str,
        op_index: int,
        collect_stats: bool,
    ) -> None:
        self.stage_index = stage_index
        self.stage_name = stage_name
        self.op_index = op_index
        self.collect_stats = collect_stats
