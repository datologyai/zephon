# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Internal testing operator that induces small deterministic delays.

This op is intended for stress-testing the threaded runner's ordering and
buffering guarantees. It computes a tiny sleep duration from each sample's
local index within its shard so that delays are deterministic and bounded.
"""

import time
from typing import Optional, TypeVar

from zephon.core.constants import SampleRecord
from zephon.core.op_base import DefaultFinalize, DefaultSetup
from zephon.core.traits import Buffering, OpTraits

T = TypeVar("T")


class DelayById(DefaultSetup, DefaultFinalize[T]):
    """Sleep a small, deterministic amount based on the sample's local id."""

    def __init__(
        self, *, max_delay_ms: float = 2.0, buffering: Optional[Buffering] = None
    ) -> None:
        DefaultSetup.__init__(self)
        self.max_delay_ms = float(max_delay_ms)
        self._buffering = buffering or Buffering(max_batch=32, max_latency_ms=2)

    def traits(self) -> OpTraits:
        return OpTraits(indexable=True, parallelism=8)

    def buffering(self) -> Optional[Buffering]:
        return self._buffering

    def process_one(self, elem: T) -> list[T]:
        return self.process_many([elem])

    def process_many(self, elems: list[T]) -> list[T]:
        out: list[T] = []
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
