# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Public stats type for the MTP hand-off queue."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MTPQueueStats:
    """Point-in-time occupancy of the MTP hand-off queue (``data_q``).

    ``depth`` counts queued items, which may not have been pickled yet;
    ``staged_bytes`` is the already-pickled data in the kernel transport
    buffer, readable without the feeder running.  Zero staged with ``depth``
    near ``capacity`` means the feeder is starved.  ``prefetch_depth`` counts
    items already drained into the main-process prefetch buffer; it is
    sampled without locking and may be slightly stale.

    A field is -1 when unmeasurable: ``depth`` on macOS (``sem_getvalue`` is
    unsupported there), ``staged_bytes`` once the queue is closed.
    """

    depth: int
    """Items put by the subprocess but not yet retrieved by the consumer."""

    capacity: int
    """Maximum ``depth`` — the resolved ``mtp_buffer``."""

    staged_bytes: int
    """Pickled bytes staged in the transport buffer, consumer-ready."""

    prefetch_depth: int
    """Deserialized items in the prefetch buffer (0 when ``mtp_prefetch=0``)."""


__all__ = ["MTPQueueStats"]
