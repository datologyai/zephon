# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Canonical data model shared across the core data-loading pipeline."""

from dataclasses import dataclass, field
from typing import Any

DatasetId = int
ShardId = int
LocalSampleId = int
SampleId = tuple[DatasetId, ShardId, LocalSampleId]


@dataclass(frozen=True)
class SampleMeta:
    """Lightweight metadata that uniquely identifies a sample in a shard."""

    sample_id: SampleId
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleRecord:
    """Sample payload bundled with its metadata for transport through stages."""

    meta: SampleMeta
    payload: dict[str, Any]


Element = Any  # TODO(MaxiBoether): better typing for this?
