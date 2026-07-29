# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Utilities that expose replay configuration to operators."""

from zephon.types import LaneId, SampleCursor


class ReplayConfigService:
    """Read-only snapshot provider for per-lane replay cursors.

    Operators use :meth:`snapshot` exactly once during setup to learn which
    prefixes must be dropped after a checkpoint restore. The engine refreshes
    the snapshot whenever a checkpoint is loaded so that future operator
    instances see the latest cursor map. No per-element communication happens
    between operators and the engine through this service.
    """

    def __init__(self) -> None:
        self._snapshot: dict[LaneId, SampleCursor | None] = {}

    def set_snapshot(self, data: dict[LaneId, SampleCursor | None]) -> None:
        """Swap the published snapshot with ``data``."""
        self._snapshot = dict(data)

    def snapshot(self) -> dict[LaneId, SampleCursor | None]:
        """Return a copy of the most recently published snapshot."""
        return dict(self._snapshot)
