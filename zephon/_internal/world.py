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


def check_step_lane_windows(
    records_per_step: int, batch_size: int, canonical_replicas: int, dp_degree: int
) -> None:
    """Raise unless every lane contributes the same whole batches to each step.

    Elastic resume keeps global batches identical only when an optimizer step
    takes an equal number of batches from every canonical lane, and every DP
    group owns the same number of lanes.
    """
    batches, remainder = divmod(records_per_step, batch_size)
    if remainder:
        raise ValueError(
            f"global_batch_size={records_per_step} must be a multiple of the "
            f"pipeline's output batch size ({batch_size})."
        )
    if batches % canonical_replicas == 0 and canonical_replicas % dp_degree == 0:
        return
    fits = [
        replicas
        for replicas in range(dp_degree, batches + 1, dp_degree)
        if batches % replicas == 0
    ]
    below = [replicas for replicas in fits if replicas <= canonical_replicas][-1:]
    above = [replicas for replicas in fits if replicas >= canonical_replicas][:1]
    options = sorted(set(below + above))
    hint = (
        "canonical_replicas=" + " or ".join(str(r) for r in options)
        if options
        else f"a global_batch_size that is a multiple of {dp_degree * batch_size}"
    )
    if (
        records_per_step % canonical_replicas == 0
        and canonical_replicas % dp_degree == 0
    ):
        per_lane = records_per_step // canonical_replicas
        batch = max(d for d in range(1, batch_size + 1) if per_lane % d == 0)
        hint += f", or an output batch size of {batch}"
    raise ValueError(
        f"An optimizer step of {records_per_step} records is {batches} batches of "
        f"{batch_size}, which does not split evenly over canonical_replicas="
        f"{canonical_replicas} lanes owned by dp_degree={dp_degree} groups: every "
        f"lane and every DP group must take an equal share. Use {hint}."
    )
