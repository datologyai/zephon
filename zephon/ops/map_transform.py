# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Map-style transformation operators for applying user-defined functions to samples."""

import logging
from typing import Callable, Optional

from zephon.core.constants import SamplePayload, SampleRecord
from zephon.core.op_base import DefaultFinalize, DefaultSetup, OpContext
from zephon.core.traits import Buffering, OpTraits

log = logging.getLogger(__name__)


class MapTransform(DefaultSetup, DefaultFinalize[SampleRecord]):
    """Apply a transformation function to each sample's payload.

    This operator supports lightweight per-sample transformations on the entire payload.

    **Filtering Support:**
    If the transform function returns ``None``, the sample is dropped (filtered out).
    This preserves determinism because:
    1. Empty results still get sequence numbers in ThreadStageRunner
    2. Ordering is preserved via sequence numbers
    3. Filter decision must be deterministic (same input -> same decision)

    **Determinism Requirement:**
    The transform function MUST be deterministic - same input must always produce
    same output (including None for filtering). Non-deterministic transforms can
    break replay/checkpointing functionality.

    **Parallelism Safety:**
    With parallelism > 1, all instances must make identical filtering decisions
    for the same inputs. This is automatically satisfied if the transform function
    is deterministic.

    Example:
        >>> def preprocess(payload):
        ...     if payload.get("text") == "":
        ...         return None  # Drop empty samples
        ...     return {"text": payload["text"].upper(), "processed": True}
        ...
        >>> op = MapTransform(preprocess, drop_none=True)
        >>> record = SampleRecord(meta=..., payload={"text": "hello"})
        >>> result = op.process_one(record)
        >>> assert result[0].payload["text"] == "HELLO"
    """

    def __init__(
        self,
        transform_fn: Callable[[SamplePayload], SamplePayload | None],
        *,
        drop_none: bool = True,
        buffering: Optional[Buffering] = None,
    ) -> None:
        DefaultSetup.__init__(self)

        if not callable(transform_fn):
            raise TypeError("transform_fn must be callable")
        self.transform_fn = transform_fn
        self.drop_none = drop_none
        self._buffering = buffering or Buffering(max_batch=64, max_latency_ms=3)

    def setup(
        self,
        ctx: OpContext,
        stage_index: int,
        stage_name: str,
        op_index: int,
        collect_stats: bool,
    ) -> None:
        DefaultSetup.setup(self, ctx, stage_index, stage_name, op_index, collect_stats)

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, parallelism=4)

    def buffering(self) -> Optional[Buffering]:
        return self._buffering

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        """Transform a single sample.

        If transform_fn returns None and drop_none=True, returns empty list (filters sample).
        Otherwise, returns list with transformed sample.
        """
        # Apply transformation to entire payload
        transformed = self.transform_fn(elem.payload)

        # Handle filtering: return empty list to drop sample
        # This preserves determinism because:
        # 1. Empty results still get sequence numbers in ThreadStageRunner
        # 2. Ordering is preserved via sequence numbers
        # 3. Filter decision must be deterministic (same input -> same decision)
        if transformed is None:
            return [] if self.drop_none else [elem]

        # Reuse existing SampleRecord to avoid allocation overhead
        elem.payload = transformed
        return [elem]

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        """Transform multiple samples efficiently."""
        results: list[SampleRecord] = []
        for elem in elems:
            results.extend(self.process_one(elem))
        return results
