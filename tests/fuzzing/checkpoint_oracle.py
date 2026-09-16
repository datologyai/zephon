# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Reusable observations and oracles for checkpoint/recovery fuzz tests."""

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from zephon import Pipeline, SampleBatch, SampleRecord
from zephon.types import SampleCursorKey, SampleId


@dataclass(frozen=True)
class ObservedContributor:
    """Contributor identity and whether this observation closes its offset."""

    cursor: SampleCursorKey
    is_last_child: bool


@dataclass(frozen=True)
class ObservedRecord:
    """Stable, comparison-friendly identity for one delivered record."""

    lane_id: int
    cursor: SampleCursorKey
    contributors: tuple[ObservedContributor, ...]
    payload: str

    @property
    def sample_id(self) -> SampleId:
        """Return the source sample identity embedded in the cursor."""
        return self.cursor[3]


ObservedWindow = tuple[ObservedRecord, ...]


def observe_item(item: SampleRecord | SampleBatch) -> ObservedWindow:
    """Convert a delivered record or batch into stable identity metadata."""
    records = (item,) if isinstance(item, SampleRecord) else item.records
    observed: list[ObservedRecord] = []
    for record in records:
        payload = record.payload
        if not isinstance(payload, dict):
            raise AssertionError(
                f"checkpoint oracle expected a dict payload, got {type(payload).__name__}"
            )
        observed.append(
            ObservedRecord(
                lane_id=int(record.meta.lane_id),
                cursor=record.meta.cursor.as_key(),
                contributors=tuple(
                    ObservedContributor(
                        cursor=ref.cursor.as_key(),
                        is_last_child=bool(ref.is_last_child),
                    )
                    for ref in record.meta.contribution_refs()
                ),
                payload=json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=repr,
                ),
            )
        )
    return tuple(observed)


def consume_windows(
    pipeline: Pipeline,
    *,
    limit: int | None = None,
    checkpoint: bool = False,
) -> tuple[list[ObservedWindow], dict[str, Any] | None]:
    """Consume delivered windows, optionally checkpointing at an exact cut."""
    windows: list[ObservedWindow] = []
    state: dict[str, Any] | None = None
    iterator = iter(pipeline)
    try:
        for item in iterator:
            windows.append(observe_item(item))
            if limit is not None and len(windows) >= limit:
                if checkpoint:
                    state = pipeline.checkpoint()
                break
    finally:
        iterator.close()
    return windows, state


def assert_source_ids_delivered_once(
    windows: list[ObservedWindow], expected: list[SampleId]
) -> None:
    """Assert that a one-to-one pipeline delivered every source sample once."""
    delivered = [record.sample_id for window in windows for record in window]
    assert Counter(delivered) == Counter(expected)

    assert_source_contributors_closed_once(windows, expected)


def assert_source_contributors_closed_once(
    windows: list[ObservedWindow], expected: list[SampleId]
) -> None:
    """Assert every source offset is closed once, including packed outputs."""

    closing = [
        contributor.cursor[3]
        for window in windows
        for record in window
        for contributor in record.contributors
        if contributor.is_last_child
    ]
    assert Counter(closing) == Counter(expected)


def assert_same_record_multiset(
    actual: list[ObservedWindow], expected: list[ObservedWindow]
) -> None:
    """Assert payload, cursor, contributor, and lane equality without ordering."""
    actual_records = [record for window in actual for record in window]
    expected_records = [record for window in expected for record in window]
    assert Counter(actual_records) == Counter(expected_records)
