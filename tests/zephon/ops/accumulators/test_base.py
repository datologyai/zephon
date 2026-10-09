# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Default accumulator handling of source-exhaustion notifications."""

from collections.abc import Sequence

from zephon.ops.accumulators import Accumulator, ReadyBatch


class _Buffering(Accumulator[int]):
    def __init__(self) -> None:
        self._buf: list[int] = []

    def push_many(self, elems: Sequence[int]) -> list[ReadyBatch[int]]:
        self._buf.extend(elems)
        return []

    def flush(
        self, *, reset: bool = False, lane_id: int | None = None
    ) -> list[ReadyBatch[int]]:
        out, self._buf = self._buf, []
        return [(out, 0)] if out else []

    def has_pending_data(self, lane_id: int | None = None) -> bool:
        return bool(self._buf)


def test_default_source_exhausted_hook_is_noop() -> None:
    acc = _Buffering()
    acc.push_many([1, 2, 3])
    assert acc.on_source_exhausted(0, 0) == []
    assert acc.has_pending_data()
    assert acc.flush() == [([1, 2, 3], 0)]
