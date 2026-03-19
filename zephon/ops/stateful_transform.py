# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Stateful transform operator for custom accumulation logic."""

from typing import Any, Callable, Generic, Optional, Sequence, TypeVar

from zephon.core.accumulators.base import Accumulator, ReadyBatch
from zephon.core.constants import SampleRecord
from zephon.core.op_base import DefaultSetup
from zephon.core.traits import OpTraits

S = TypeVar("S")  # State type


class StatefulTransformAccumulator(Accumulator[SampleRecord], Generic[S]):
    """Accumulator that wraps user-provided state management functions.

    This accumulator runs on the pump thread (serial) and allows users to
    implement custom buffering and batching logic without understanding the
    full Op protocol complexity.

    The state lifecycle:
    1. State is lazily initialized on first push_many() call via init_state()
    2. Each push_many() calls push_fn(state, items) -> (new_state, outputs)
    3. If should_flush_fn returns True, flush_fn is called and state is reset
    4. On stream end, flush() emits any remaining buffered items
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
        self._state: S | None = None
        self._initialized = False
        self._flushed = False
        self._has_pending = False

    def push_many(
        self, items: Sequence[SampleRecord]
    ) -> list[ReadyBatch[SampleRecord]]:
        if not self._initialized:
            self._state = self._init_state()
            self._initialized = True

        result: list[ReadyBatch[SampleRecord]] = []
        items_list = items if isinstance(items, list) else list(items)
        if not items_list:
            return result
        if self._flush_fn is not None:
            self._has_pending = True
        self._state, outputs = self._push_fn(self._state, items_list)

        if self._should_flush_fn and self._should_flush_fn(self._state):
            flush_outputs = self._flush_fn(self._state) if self._flush_fn else []
            self._state = self._init_state()
            self._has_pending = False
            outputs.extend(flush_outputs)

        if outputs:
            result.append((outputs, len(outputs)))
        return result

    def flush(self, *, reset: bool = False) -> list[ReadyBatch[SampleRecord]]:
        """Emit any remaining buffered items.

        Args:
            reset: If True (epoch boundary), re-initialize state so the
                accumulator is ready for the next epoch. If False (default),
                this is the final flush — mark as done and discard state.
        """
        result: list[ReadyBatch[SampleRecord]] = []
        if self._flush_fn and self._initialized and self._state is not None:
            outputs = self._flush_fn(self._state)
            if outputs:
                result.append((outputs, len(outputs)))

        if not reset:
            # Final flush — mark as done, discard state.
            self._flushed = True
            self._state = None
        else:
            # Mid-stream flush — re-initialize for the next epoch.
            # _initialized stays True (init_state already called).
            # _flushed stays False (accumulator continues to accept data).
            self._state = self._init_state()
        self._has_pending = False

        return result

    def has_pending_data(self) -> bool:
        """Return True if the accumulator may still need a flush."""
        if self._flushed:
            return False
        if not self._initialized or self._state is None:
            return False
        return self._has_pending


class StatefulTransformOp(DefaultSetup, Generic[S]):
    """Operator that wraps user-provided stateful transformation logic.

    This provides a higher-level API for implementing custom accumulators
    without requiring users to understand the full Op protocol. Users only
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
        DefaultSetup.__init__(self)
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
