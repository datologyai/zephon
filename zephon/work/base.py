# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Abstract base definitions for work sources and chunks."""

from dataclasses import dataclass
from typing import Protocol

from zephon.core.constants import SampleId


# TODO(MaxiBoether): have better impl of worksource that e.g. has iterator and offers sub chunking.
@dataclass
class WorkChunk:
    """Bundle of sample identifiers handed to the engine for processing."""

    sample_ids: list[SampleId]
    seed: int


class WorkSource(Protocol):
    """Protocol for producing work chunks and supporting random access."""

    def next_chunk(self) -> WorkChunk | None: ...

    def checkpoint(self) -> bytes: ...

    def restore(self, state: bytes) -> None: ...

    def supports_indexing(self) -> bool: ...

    def __len__(self) -> int: ...

    def sample_id_at(self, index: int) -> SampleId: ...
