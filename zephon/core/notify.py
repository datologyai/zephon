# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Notification helpers shared by inline and MTP iteration paths.

Extract lightweight notification args from stream items and dispatch
them to the engine.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from zephon.core.constants import (
    ContributorRef,
    SampleBatch,
    SampleCursor,
    SampleRecord,
    StreamItem,
)


class _MonotoneNotify(NamedTuple):
    """Lightweight args for ``engine.notify_monotone()``."""

    lane_id: int
    max_chunk_id: int
    add_k: int
    max_cursor: SampleCursor | None


class _ContributorNotify(NamedTuple):
    """Lightweight args for ``engine.notify()``."""

    lane_id: int
    entries: list[ContributorRef]
    record_cursor: SampleCursor | None


NotifyArgs = _MonotoneNotify | _ContributorNotify


def _extract_notify_args(item: StreamItem, use_monotone: bool) -> NotifyArgs:
    """Extract lightweight notification args from a stream item.

    Returns a small NamedTuple of scalars / cursor refs — never retains
    the full item payload.  Used by the subprocess to populate
    ``pending[seq]`` without holding large batches in memory.
    """
    if isinstance(item, SampleBatch):
        if not item.records:
            raise TypeError("SampleBatch must contain at least one record")
        lane_id = item.lane_ids[0]
        if use_monotone:
            chunk_ids = item.chunk_ids
            max_cid = chunk_ids[0]
            n_cursors = 0
            max_cursor: SampleCursor | None = None
            for i, r in enumerate(item.records):
                cid = chunk_ids[i]
                if cid > max_cid:
                    max_cid = cid
                    n_cursors = 1
                    max_cursor = r.meta.cursor
                elif cid == max_cid:
                    n_cursors += 1
                    c = r.meta.cursor
                    if max_cursor is None or c > max_cursor:
                        max_cursor = c
            return _MonotoneNotify(lane_id, max_cid, n_cursors, max_cursor)
        contributors: list[ContributorRef] = []
        record_cursor: SampleCursor | None = item.records[-1].meta.cursor
        for record in item.records:
            contributors.extend(record.meta.contribution_refs())
        return _ContributorNotify(lane_id, contributors, record_cursor)

    if isinstance(item, SampleRecord):
        lane_id = item.meta.lane_id
        if use_monotone:
            return _MonotoneNotify(lane_id, item.meta.chunk_id, 1, item.meta.cursor)
        return _ContributorNotify(
            lane_id, item.meta.contribution_refs(), item.meta.cursor
        )

    raise TypeError(
        f"Unsupported element type: {type(item)!r}; "
        "expected SampleBatch or SampleRecord"
    )


def _apply_notify_args(engine: Any, notify: NotifyArgs) -> None:
    """Dispatch pre-extracted notify args to the engine."""
    if isinstance(notify, _MonotoneNotify):
        engine.notify_monotone(
            notify.lane_id, notify.max_chunk_id, notify.add_k, notify.max_cursor
        )
    else:
        engine.notify(
            notify.lane_id, notify.entries, record_cursor=notify.record_cursor
        )


def _notify_item(engine: Any, item: StreamItem, use_monotone: bool) -> None:
    """Extract + apply in one step.  Used by the inline iteration path."""
    _apply_notify_args(engine, _extract_notify_args(item, use_monotone))


def is_tombstone(item: StreamItem) -> bool:
    """Check if a stream item is a tombstone (should be notified but not yielded)."""
    return isinstance(item, SampleRecord) and item.meta.tombstone


def is_sentinel(item: StreamItem) -> bool:
    """Check if a stream item is a sentinel (should be notified but not yielded)."""
    return isinstance(item, SampleRecord) and item.meta.is_sentinel
