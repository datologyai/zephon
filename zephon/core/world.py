"""Canonical world and lane scheduling state for elastic determinism."""

from dataclasses import dataclass, field


@dataclass
class World:
    """Canonical scheduling world for elastic runs.

    Attributes:
        canonical_replicas: Number of canonical replicas (logical DP) defining
            the schedule. Internally represented by lanes.
        mapping: Physical rank to list of owned canonical lane IDs.
    """

    canonical_replicas: int = 1
    worker_id: int = 0
    workers_per_rank: int = 1
    physical_rank: int = 0
    num_ranks: int = 1
    lanes_for_rank: dict[int, list[int]] = field(default_factory=lambda: {0: [0]})
