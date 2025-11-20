# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Abstract operator contracts shared by the planner and runtime."""

from typing import Any, Generic, Optional, Protocol, TypeVar

from zephon.core.constants import StreamItem
from zephon.core.traits import Buffering, OpTraits


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

    def buffering(self) -> Optional[Buffering]: ...

    # Concrete ops specify precise input/output element types.
    def process_one(self, elem: InT) -> list[OutT]: ...

    def process_many(self, elems: list[InT]) -> list[OutT]: ...

    def finalize(self) -> list[OutT]:
        """Emit any buffered outputs once the upstream iterator is exhausted."""
        ...


class DefaultFinalize(Generic[OutT]):
    """Mixin providing a no-op ``finalize`` implementation."""

    def finalize(self) -> list[OutT]:
        """Return an empty list when the operator has no buffered tail."""
        return []


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
