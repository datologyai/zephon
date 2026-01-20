# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Internal testing operator that induces small deterministic delays.

This op is intended for stress-testing the threaded runner's ordering and
buffering guarantees. It computes a tiny sleep duration from each sample's
local index within its shard so that delays are deterministic and bounded.
"""

import time
from typing import Optional

from zephon.core.accumulators import Accumulator, CountingAccumulator
from zephon.core.constants import SampleRecord, StreamItem
from zephon.core.op_base import DefaultSetup
from zephon.core.traits import OpTraits


class DelayById(DefaultSetup):
    """Sleep a small, deterministic amount based on the sample's local id."""

    def __init__(
        self,
        *,
        max_delay_ms: float = 2.0,
        max_batch: int = 32,
        max_latency_ms: Optional[int] = 2,
    ) -> None:
        DefaultSetup.__init__(self)
        self.max_delay_ms = float(max_delay_ms)
        self._max_batch = max_batch
        self._max_latency_ms = max_latency_ms

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, preserves_cursor_order=True, parallelism=8)

    def accumulator(self, *, deterministic: bool) -> Accumulator[StreamItem]:
        return CountingAccumulator[StreamItem](
            max_batch=self._max_batch,
            max_latency_ms=None if deterministic else self._max_latency_ms,
        )

    def process_one(self, elem: StreamItem) -> list[StreamItem]:
        return self.process_many([elem])

    def process_many(self, elems: list[StreamItem]) -> list[StreamItem]:
        out: list[StreamItem] = []
        slots = 5  # map ids/hashes into 0..4
        for item in elems:
            # Derive a stable bucket from either the SampleRecord's local id
            # or a generic hash for non-record elements
            try:
                if isinstance(item, SampleRecord):
                    _, _, local = item.meta.sample_id
                    bucket = abs(int(local)) % slots
                else:
                    bucket = abs(int(hash(item))) % slots
            except Exception:
                bucket = 0
            delay_sec = (self.max_delay_ms * (bucket / max(1, slots - 1))) / 1000.0
            if delay_sec > 0:
                time.sleep(delay_sec)
            out.append(item)
        return out


__all__ = ["DelayById"]
