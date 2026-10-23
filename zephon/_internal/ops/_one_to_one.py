# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""One-source-record-per-example contract for optimizer-step curricula.

A step plan maps source records to optimizer steps by count, so dropping,
adding, or replacing a record shifts every later transition. Data-dependent
drops are checked where they happen, before a tombstone is emitted.
"""

from typing import Any

from zephon.types import SampleRecord

#: Execution-context key set while a ``StepMixturePlan`` drives the source.
REQUIRE_ONE_TO_ONE_RECORDS = "require_one_to_one_records"

RecordIdentity = tuple[Any, ...]


def record_identity(record: SampleRecord) -> RecordIdentity:
    meta = record.meta
    return (meta.lane_id, meta.chunk_id, meta.chunk_offset, meta.sample_id)


def one_to_one_violation(op: str, what: str, at: RecordIdentity) -> ValueError:
    lane_id, chunk_id, chunk_offset, _ = at
    return ValueError(
        f"StepMixturePlan requires one source record per training example, but "
        f"{op} {what} at lane {lane_id}, chunk {chunk_id}, offset {chunk_offset}. "
        "Filter the dataset before loading, or use a chunk-round MixtureSchedule."
    )
