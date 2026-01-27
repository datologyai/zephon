# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Lane-aware replay filter that drops the already-consumed prefix.

This operator runs immediately before batching (when present) or as the tail
of the pipeline when no batching is used.
"""

from typing import Any

from zephon.core.accumulators import Accumulator, PassthroughAccumulator
from zephon.core.constants import LaneId, SampleCursor, SampleRecord
from zephon.core.op_base import DefaultSetup, OpContext
from zephon.core.replay import ReplayConfigService
from zephon.core.traits import OpTraits


class ReplayFilter(DefaultSetup):
    """Per-lane pre-batch dropper configured lazily via the OpContext.

    Drops records until the checkpointed per-lane cursor reappears (inclusive),
    then emits the suffix unchanged. Assumes it runs single-threaded within its
    stage so that per-lane state is not sharded across threads.
    """

    def __init__(self) -> None:
        DefaultSetup.__init__(self)

        self._service: ReplayConfigService | None = None
        self._targets: dict[LaneId, SampleCursor | None] = {}
        self._seen_target: dict[LaneId, bool] = {}
        self._disabled: bool = True
        self._initialized = False

    def setup(
        self,
        ctx: OpContext,
        stage_index: int,
        stage_name: str,
        op_index: int,
        collect_stats: bool,
    ) -> None:
        DefaultSetup.setup(self, ctx, stage_index, stage_name, op_index, collect_stats)

        service = ctx.get("replay_state_service")
        if service is None:
            raise RuntimeError("ReplayConfigService not available!")
        self._service = service
        self._disabled = False

    def traits(self) -> OpTraits:
        return OpTraits(
            indexable=True,
            preserves_cursor_order=True,
            batch_shape_sensitive=False,
        )

    def accumulator(
        self, *, deterministic: bool, ctx: dict[str, Any]
    ) -> Accumulator[SampleRecord]:
        return PassthroughAccumulator[SampleRecord]()

    def _init_replay(self) -> None:
        if self._initialized:
            return
        assert self._service is not None

        self._initialized = True
        snapshot = self._service.snapshot()
        self._targets = dict(snapshot.items())
        self._seen_target = {lane: cursor is None for lane, cursor in snapshot.items()}
        if not self._targets:
            self._disabled = True
        else:
            self._disabled = all(self._seen_target.values())

    def _should_drop(self, elem: SampleRecord) -> bool:
        assert self._service is not None
        lane = int(elem.meta.lane_id)
        cursor = elem.meta.cursor
        target = self._targets.get(lane)

        if lane not in self._seen_target:
            self._seen_target[lane] = target is None

        if target is None:
            if all(self._seen_target.values()):
                self._disabled = True
            return False

        if not self._seen_target[lane]:
            if cursor == target:
                self._seen_target[lane] = True
                if all(self._seen_target.values()):
                    self._disabled = True
                return True  # drop sentinel itself
            return True  # still replay prefix

        return False

    def process_one(self, elem: SampleRecord) -> list[SampleRecord]:
        self._init_replay()
        if self._disabled:
            return [elem]
        if self._should_drop(elem):
            return []
        return [elem]

    def process_many(self, elems: list[SampleRecord]) -> list[SampleRecord]:
        self._init_replay()
        if self._disabled:
            return elems
        out: list[SampleRecord] = []
        for e in elems:  # TODO: can we vectorize?
            if not self._should_drop(e):
                out.append(e)
        # If nothing was dropped, return original list to avoid copy overhead
        # TODO: we already materialized out though, we can avoid that by first collecting the resutls of should drop
        return elems if len(out) == len(elems) else out
