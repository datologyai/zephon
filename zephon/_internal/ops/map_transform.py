# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Map-style transformation operators for applying user-defined functions to samples or batches."""

import logging
from typing import Any, Callable, Optional

from zephon.ops.accumulators import Accumulator, CountingAccumulator
from zephon.ops.base import BaseOp
from zephon.ops.children import tombstones_for_record
from zephon.ops.traits import OpTraits
from zephon.types import SampleBatch, SamplePayload, SampleRecord

log = logging.getLogger(__name__)


class _BaseMapTransform(BaseOp):
    """Shared base for map-style transformation operators.

    Subclasses specialise ``process_one`` for either individual samples
    (``MapTransform``) or batches (``MapBatchTransform``).
    """

    def __init__(
        self,
        transform_fn: Callable,
        *,
        drop_none: bool = True,
        max_batch: int = 64,
        max_latency_ms: Optional[int] = 3,
    ) -> None:
        super().__init__()

        if not callable(transform_fn):
            raise TypeError("transform_fn must be callable")
        self.transform_fn = transform_fn
        self.drop_none = drop_none
        self._max_batch = max_batch
        self._max_latency_ms = max_latency_ms

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, preserves_cursor_order=True, parallelism=4)

    def _tombstones_for(self, elem: SampleRecord | SampleBatch) -> list[SampleRecord]:
        """Emit tombstone records for every closing contributor in *elem*."""
        records = (elem,) if isinstance(elem, SampleRecord) else elem.records
        tombstones: list[SampleRecord] = []
        for record in records:
            tombstones.extend(tombstones_for_record(record))
        return tombstones


class MapTransform(_BaseMapTransform):
    """Apply a transformation function to individual sample payloads.

    This operator works on ``SampleRecord`` inputs only and must be placed
    **before** a ``Batch`` operator in the pipeline.  For transformations on
    batched data, use ``MapBatchTransform`` (exposed as ``pipeline.map_batch()``).

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
        max_batch: int = 64,
        max_latency_ms: Optional[int] = 3,
    ) -> None:
        _BaseMapTransform.__init__(
            self,
            transform_fn,
            drop_none=drop_none,
            max_batch=max_batch,
            max_latency_ms=max_latency_ms,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return CountingAccumulator[SampleRecord](
            max_batch=self._max_batch,
            max_latency_ms=None if deterministic else self._max_latency_ms,
        )

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        """Transform a sample's payload.

        When ``drop_none=True`` and the transform returns ``None``, tombstone
        records are emitted to properly close chunk offsets for the dropped item.
        Tombstone records pass through unchanged.
        """
        # Tombstones pass through unchanged — they carry no payload to transform.
        if elem.meta.tombstone:
            return [elem]

        transformed = self.transform_fn(elem.payload)

        if transformed is None:
            if not self.drop_none:
                return [elem]
            return self._tombstones_for(elem)

        elem.payload = transformed
        return [elem]

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        """Transform multiple samples."""
        results: list[SampleRecord] = []
        for elem in elems:
            results.extend(self.process_one(elem))
        return results


class MapBatchTransform(_BaseMapTransform):
    """Apply a transformation function to batches of samples.

    This operator works on ``SampleBatch`` inputs only and must be placed
    **after** a ``Batch`` operator in the pipeline.  For transformations on
    individual samples, use ``MapTransform`` (exposed as
    ``pipeline.map_transform()``).

    The transform function receives the entire ``SampleBatch`` and should
    return a (possibly modified) ``SampleBatch``, or ``None`` to drop it.

    **Filtering Support:**
    If the transform function returns ``None``, the batch is dropped (filtered out).
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
        >>> def process_batch(batch):
        ...     # Transform receives the entire SampleBatch
        ...     return batch  # or return modified batch
        ...
        >>> op = MapBatchTransform(process_batch)
        >>> batch = SampleBatch(records=(...))
        >>> result = op.process_one(batch)
        >>> assert isinstance(result[0], SampleBatch)
    """

    def __init__(
        self,
        transform_fn: Callable[[SampleBatch], SampleBatch | None],
        *,
        drop_none: bool = True,
        max_batch: int = 64,
        max_latency_ms: Optional[int] = 3,
    ) -> None:
        _BaseMapTransform.__init__(
            self,
            transform_fn,
            drop_none=drop_none,
            max_batch=max_batch,
            max_latency_ms=max_latency_ms,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord | SampleBatch]:
        # Accepts SampleRecord | SampleBatch because the upstream Batch operator
        # emits tombstone SampleRecords alongside SampleBatches.
        return CountingAccumulator[SampleRecord | SampleBatch](
            max_batch=self._max_batch,
            max_latency_ms=None if deterministic else self._max_latency_ms,
        )

    def process_one(self, elem: SampleBatch) -> list[SampleBatch | SampleRecord]:
        """Transform a batch.

        When ``drop_none=True`` and the transform returns ``None``, tombstone
        records are emitted for every record in the batch.
        """
        transformed = self.transform_fn(elem)

        if transformed is None:
            if not self.drop_none:
                return [elem]
            # _tombstones_for returns list[SampleRecord]; widen via extend.
            out: list[SampleBatch | SampleRecord] = []
            out.extend(self._tombstones_for(elem))
            return out

        return [transformed]

    def process_many(
        self, elems: list[SampleRecord | SampleBatch]
    ) -> list[SampleBatch | SampleRecord]:
        """Transform multiple batches.

        Tombstone ``SampleRecord`` objects from the upstream ``Batch`` operator
        are passed through unchanged; only ``SampleBatch`` elements are
        forwarded to ``process_one``.
        """
        results: list[SampleBatch | SampleRecord] = []
        for elem in elems:
            if isinstance(elem, SampleRecord):
                # Tombstone records from Batch pass through unchanged.
                results.append(elem)
            else:
                results.extend(self.process_one(elem))
        return results
