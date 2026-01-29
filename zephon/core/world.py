"""Canonical world and lane scheduling state for elastic determinism."""

from dataclasses import dataclass, field


@dataclass
class World:
    """Canonical scheduling world for elastic runs.

    Attributes:
        canonical_replicas: Number of canonical replicas (logical DP) defining
            the schedule. Internally represented by lanes.

        # Global coordination
        world_size: Total number of ranks (GPUs).
        global_rank: Unique identifier for this rank.

        # Data partitioning
        dp_degree: Number of data parallel groups.
        dp_group_id: Which data partition this rank reads.
        lanes_for_dp_group: Mapping from dp_group_id to owned lanes.

        # Worker info
        worker_id: DataLoader worker ID (0 if no workers).
        workers_per_rank: DataLoader workers per rank (1 if no workers).
    """

    canonical_replicas: int = 1

    # Global coordination
    world_size: int = 1
    global_rank: int = 0

    # Data partitioning
    dp_degree: int = 1
    dp_group_id: int = 0
    lanes_for_dp_group: dict[int, list[int]] = field(default_factory=lambda: {0: [0]})

    # Worker info
    worker_id: int = 0
    workers_per_rank: int = 1
