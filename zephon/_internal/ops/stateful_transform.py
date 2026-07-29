# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Stateful transform operator for custom accumulation logic."""

from typing import Any, Callable, Generic, Optional, Sequence, TypeVar

from zephon.ops.accumulators.base import Accumulator, ReadyBatch
from zephon.ops.base import BaseOp
from zephon.ops.traits import OpTraits
from zephon.types import SampleRecord

S = TypeVar("S")  # State type


class StatefulTransformAccumulator(Accumulator[SampleRecord], Generic[S]):
    """Accumulator that wraps user-provided state management functions.

    This accumulator runs on the pump thread (serial) and allows users to
    implement custom buffering and batching logic without writing a complete
    ``BaseOp`` subclass.

    State is partitioned by lane, like every other stateful accumulator: each
    lane gets its own ``init_state()`` instance and ``push_fn`` sees one lane's
    records at a time.  Lanes are independent streams, so a flush sentinel for
    one lane resets only that lane's state — required for deterministic replay
    across checkpoint/restore when one engine owns several lanes.

    The per-lane state lifecycle:
    1. A lane's state is created lazily on its first ``push_many()`` element.
    2. Each ``push_many()`` groups elements by lane and calls
       ``push_fn(state, lane_items) -> (new_state, outputs)`` per lane.
    3. If ``should_flush_fn`` returns True for a lane, ``flush_fn`` runs and
       that lane's state resets.
    4. ``flush(lane_id=...)`` drains one lane; ``flush()`` (stream end) drains
       every lane.
    """

    def __init__(
        self,
        init_state: Callable[[], S],
        push_fn: Callable[[S, list[SampleRecord]], tuple[S, list[SampleRecord]]],
        flush_fn: Optional[Callable[[S], list[SampleRecord]]],
        should_flush_fn: Optional[Callable[[S], bool]],
    ):
        self._init_state = init_state
        self._push_fn = push_fn
        self._flush_fn = flush_fn
        self._should_flush_fn = should_flush_fn
        # One user state per lane, created lazily on that lane's first element.
        self._states: dict[int, S] = {}
        # Lanes that have taken data since their last flush.  Only tracked when
        # a flush_fn exists, since without one flush() emits nothing.
        self._pending: set[int] = set()
        self._flushed = False

    # TODO: StatefulTransformAccumulator delegates to a user-supplied push_fn
    # which *may* read payload. For now we default to False (no eager resolution)
    # so that metadata-only push_fns don't pay the cost. If a push_fn touches
    # payload it will fail on a LazyPayload — the caller should set
    # reads_payload=True at construction time or resolve manually.

    def push_many(
        self, items: Sequence[SampleRecord]
    ) -> list[ReadyBatch[SampleRecord]]:
        result: list[ReadyBatch[SampleRecord]] = []
        if not items:
            return result

        # Fold each lane's records into that lane's own state, so push_fn never
        # mixes lanes and a later per-lane flush can reset one lane in isolation.
        by_lane: dict[int, list[SampleRecord]] = {}
        for item in items:
            by_lane.setdefault(item.meta.lane_id, []).append(item)

        for lane_id, lane_items in by_lane.items():
            state = self._states.get(lane_id)
            if state is None:
                state = self._init_state()
            if self._flush_fn is not None:
                self._pending.add(lane_id)
            state, outputs = self._push_fn(state, lane_items)

            if self._should_flush_fn and self._should_flush_fn(state):
                flush_outputs = self._flush_fn(state) if self._flush_fn else []
                state = self._init_state()
                self._pending.discard(lane_id)
                outputs.extend(flush_outputs)

            self._states[lane_id] = state
            if outputs:
                result.append((outputs, len(outputs)))
        return result

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[SampleRecord]]:
        """Emit any remaining buffered items.

        Flush sentinels are per-lane: when ``lane_id`` is set only that lane's
        state is flushed and reset, leaving other lanes' state for their own
        sentinels.  ``lane_id=None`` flushes every lane (stream end).

        Args:
            reset: True at an epoch boundary — re-initialize the flushed lane(s)
                for the next epoch.  False (default) is the final flush, which
                discards state.
            lane_id: Flush only this lane when set; all lanes when None.
        """
        if lane_id is None:
            lanes = sorted(self._states)
        else:
            lanes = [lane_id] if lane_id in self._states else []

        result: list[ReadyBatch[SampleRecord]] = []
        for lid in lanes:
            if self._flush_fn is not None:
                outputs = self._flush_fn(self._states[lid])
                if outputs:
                    result.append((outputs, len(outputs)))
            self._pending.discard(lid)
            if reset:
                self._states[lid] = self._init_state()
            else:
                del self._states[lid]

        # The all-lanes, non-reset flush is the terminal stream-end drain.
        if lane_id is None and not reset:
            self._flushed = True
        return result

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        """Return True if a flush would still emit buffered data.

        Scoped to ``lane_id`` when set; otherwise True if any lane is pending.
        """
        if self._flushed:
            return False
        if lane_id is None:
            return bool(self._pending)
        return lane_id in self._pending


class StatefulTransformOp(BaseOp, Generic[S]):
    """Operator that wraps user-provided stateful transformation logic.

    This provides a higher-level API for implementing custom accumulators
    without requiring users to write a complete ``BaseOp`` subclass. Users only
    need to provide:

    - init_state: Factory that creates initial state
    - push_fn: Process items and update state, returning outputs
    - flush_fn: (optional) Emit remaining items at end-of-stream
    - should_flush_fn: (optional) Trigger early flush based on state
    - transform_fn: (optional) Batch-level transform that runs in parallel workers

    The operator handles all the protocol complexity (setup, traits,
    accumulator lifecycle) automatically.

    Execution Model:
    - push_fn/flush_fn run on the pump thread (serial) for state management
    - transform_fn runs in parallel workers for expensive computation
    - Batch structure from the accumulator is preserved for transform_fn

    This split allows patterns like "deduplicate (serial) then encode (parallel)"
    while preserving batch structure for efficient GPU processing.

    Example - Custom batching by token count:
        >>> op = StatefulTransformOp(
        ...     init_state=lambda: {"buffer": [], "tokens": 0},
        ...     push_fn=lambda s, items: accumulate_by_tokens(s, items, max_tokens=4096),
        ...     flush_fn=lambda s: s["buffer"] if s["buffer"] else [],
        ... )

    Example - Deduplicate (serial) then batch encode (parallel):
        >>> op = StatefulTransformOp(
        ...     init_state=lambda: set(),
        ...     push_fn=lambda seen, items: (
        ...         seen | {i.payload["id"] for i in items},
        ...         [i for i in items if i.payload["id"] not in seen]
        ...     ),
        ...     transform_fn=lambda batch: batch_encode(batch),  # process whole batch
        ...     parallelism=8,
        ... )
    """

    def __init__(
        self,
        init_state: Callable[[], S],
        push_fn: Callable[[S, list[SampleRecord]], tuple[S, list[SampleRecord]]],
        flush_fn: Optional[Callable[[S], list[SampleRecord]]] = None,
        should_flush_fn: Optional[Callable[[S], bool]] = None,
        transform_fn: Optional[
            Callable[[list[SampleRecord]], list[SampleRecord]]
        ] = None,
        parallelism: int = 1,
        indexable: bool = False,
        preserves_cursor_order: bool = True,
    ):
        super().__init__()
        if not callable(init_state):
            raise TypeError("init_state must be callable")
        if not callable(push_fn):
            raise TypeError("push_fn must be callable")
        if flush_fn is not None and not callable(flush_fn):
            raise TypeError("flush_fn must be callable")
        if should_flush_fn is not None and not callable(should_flush_fn):
            raise TypeError("should_flush_fn must be callable")
        if transform_fn is not None and not callable(transform_fn):
            raise TypeError("transform_fn must be callable")

        self._init_state = init_state
        self._push_fn = push_fn
        self._flush_fn = flush_fn
        self._should_flush_fn = should_flush_fn
        self._transform_fn = transform_fn
        self._parallelism = parallelism
        self._indexable = indexable
        self._preserves_cursor_order = preserves_cursor_order

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=self._indexable,
            preserves_cursor_order=self._preserves_cursor_order,
            parallelism=self._parallelism,
            batch_shape_sensitive=True,  # State depends on batch boundaries
            # Note: requires_serial_state=False because the accumulator already
            # runs serially on the pump thread. The transform_fn in workers is
            # stateless and can safely parallelize.
            requires_serial_state=False,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return StatefulTransformAccumulator(
            init_state=self._init_state,
            push_fn=self._push_fn,
            flush_fn=self._flush_fn,
            should_flush_fn=self._should_flush_fn,
        )

    # TODO: StatefulTransform does not auto-emit tombstones when push_fn or
    # transform_fn drops items. Unlike MapTransform (which emits tombstones
    # for drops via _tombstones_for), items silently filtered here will have
    # their chunk offsets never closed — a memory leak in the general notify
    # path. Investigate adding tombstone support, e.g. by tracking input vs
    # output records in process_many and emitting tombstones for the diff.

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        """Transform a single record (runs in parallel workers).

        Note: For batch-level transforms, this wraps the single item in a list,
        calls transform_fn, and unwraps. For best performance with batch-oriented
        transforms (GPU batching, etc.), use process_many.
        """
        if self._transform_fn is None:
            return [elem]
        return self._transform_fn([elem])

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        """Transform a batch of records (runs in parallel workers).

        Preserves batch structure from the accumulator, enabling efficient
        batch-level processing (e.g., GPU batching, vectorized operations).
        """
        if self._transform_fn is None:
            return elems
        return self._transform_fn(elems)
