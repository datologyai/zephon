# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Queue-drain layer of the concurrent stage runner hierarchy.

``QueueDrainStageRunner`` and ``QueueDrainOperatorState`` sit between the
queue-agnostic :class:`ConcurrentStageRunner` base and the concrete thread,
process, and Ray runners. They add the bounded-queue worker→pump handoff
machinery — ``result_queue`` on the operator state, ``pending_puts`` to
close the handoff-observation window, and queue-pop implementations of the
two source-specific hooks declared abstract on the base.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass, field
from typing import Generic, Sequence, TypeVar

from zephon.core.constants import RunnerStreamIn
from zephon.runners.concurrent import (
    ConcurrentOperatorState,
    ConcurrentRunContext,
    ConcurrentStageRunner,
    RunnerResult,
    StopToken,
    _InflightCounter,
    _QueueLike,
)

SQueueDrain = TypeVar("SQueueDrain", bound="QueueDrainOperatorState")


@dataclass
class QueueDrainOperatorState(ConcurrentOperatorState):
    """Operator state for runners with a worker→pump result queue handoff.

    Adds the ``result_queue`` that workers push :class:`RunnerResult`s into
    and ``pending_puts`` to close the handoff-observation window used by
    :attr:`all_completions_observed`.
    """

    pending_puts: _InflightCounter = field(init=False)
    result_queue: _QueueLike[RunnerResult] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.pending_puts = _InflightCounter()

    @property
    def all_completions_observed(self) -> bool:
        return self.pending_puts.is_zero()


class QueueDrainStageRunner(ConcurrentStageRunner[SQueueDrain], Generic[SQueueDrain]):
    """:class:`ConcurrentStageRunner` variant using a ``result_queue`` handoff.

    Thread and process runners follow this pattern: workers run outside the
    pump thread and return completions by pushing a :class:`RunnerResult`
    onto the operator's ``result_queue``. The pump reads from that queue.

    Implements the source-specific hooks declared abstract on
    :class:`ConcurrentStageRunner`:

    * :meth:`_drain_results` pops every ready result non-blockingly.
    * :meth:`_await_one_result` blocks up to ``timeout`` for a single result.

    ``FileNotFoundError`` (raised when a multiprocessing queue's writer dies
    mid-handoff, e.g. torch shared-memory fds during abrupt worker
    termination) is allowed to propagate out of these hooks; the outer
    ``_operator_loop`` ``BaseException`` handler records the error and
    exits the pump. ``_drain_results`` catches it explicitly only to stop
    draining before the termination check re-runs.
    """

    def _drain_results(
        self,
        state: SQueueDrain,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
    ) -> None:
        while True:
            try:
                item = self._queue_get_nowait(state.result_queue)
            except queue.Empty:
                break
            except FileNotFoundError as exc:
                self._record_error(context, exc)
                return
            self._handle_result(state, item, next_queue, context)

    def _await_one_result(
        self,
        state: SQueueDrain,
        next_queue: _QueueLike[Sequence[RunnerStreamIn] | StopToken] | None,
        context: ConcurrentRunContext,
        timeout: float,
    ) -> None:
        try:
            item = self._queue_get(state.result_queue, timeout=timeout)
        except queue.Empty:
            return
        # FileNotFoundError propagates — outer _operator_loop BaseException
        # handler records the error and returns from the pump, matching the
        # pre-refactor direct-return behavior.
        self._handle_result(state, item, next_queue, context)
