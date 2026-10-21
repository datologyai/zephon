# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Metadata shared by calibration replay and its validation probes."""

from zephon.types import SampleId, SampleMeta


def sample_meta(sample_id: SampleId) -> SampleMeta:
    """Keep source identity with placeholder scheduling and component metadata."""
    return SampleMeta(sample_id=sample_id, lane_id=0, chunk_id=0)
