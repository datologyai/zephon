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

    Elastic resume keeps global batches identical only when every DP group owns
    the same number of lanes and an optimizer step takes an equal number of
    batches from every lane. Every suggested fix satisfies both.
    """
    batches, remainder = divmod(records_per_step, batch_size)
    if remainder:
        raise ValueError(
            f"global_batch_size={records_per_step} must be a multiple of the "
            f"pipeline's output batch size ({batch_size})."
        )
    owned_evenly = canonical_replicas % dp_degree == 0
    if owned_evenly and batches % canonical_replicas == 0:
        return

    def step_sizes(replicas: int) -> str:
        unit = replicas * batch_size
        lower = records_per_step // unit * unit
        return " or ".join(str(size) for size in (lower, lower + unit) if size > 0)

    fits = [
        replicas
        for replicas in range(dp_degree, batches + 1, dp_degree)
        if batches % replicas == 0
    ]
    below = [replicas for replicas in fits if replicas <= canonical_replicas][-1:]
    above = [replicas for replicas in fits if replicas >= canonical_replicas][:1]
    replicas_fit = sorted(set(below + above))
    options = []
    if owned_evenly:
        problem = (
            f"An optimizer step of {records_per_step} records is {batches} batches "
            f"of {batch_size}, which does not split evenly over canonical_replicas="
            f"{canonical_replicas} lanes: every lane must take an equal share."
        )
        # Keeping the layout is listed first: canonical_replicas is often chosen
        # for future DP degrees and the batch size for GPU memory.
        options.append(f"global_batch_size={step_sizes(canonical_replicas)}")
        if replicas_fit:
            options.append(
                "canonical_replicas=" + " or ".join(str(r) for r in replicas_fit)
            )
        if records_per_step % canonical_replicas == 0:
            per_lane = records_per_step // canonical_replicas
            batch = max(d for d in range(1, batch_size + 1) if per_lane % d == 0)
            options.append(f"an output batch size of {batch}")
    else:
        problem = (
            f"canonical_replicas={canonical_replicas} is not a multiple of "
            f"dp_degree={dp_degree}: every DP group must own the same number of "
            "lanes for optimizer steps to stay identical across elastic resumes."
        )
        if replicas_fit:
            options.append(
                "canonical_replicas=" + " or ".join(str(r) for r in replicas_fit)
            )
        else:
            problem += (
                f" No multiple of {dp_degree} lanes splits the {batches} batches "
                f"of {batch_size} in a step evenly, so global_batch_size must "
                "change too."
            )
            lower = canonical_replicas // dp_degree * dp_degree
            options.extend(
                f"canonical_replicas={replicas} with global_batch_size="
                f"{step_sizes(replicas)}"
                for replicas in (lower, lower + dp_degree)
                if replicas > 0
            )
    hint = (
        options[0]
        if len(options) == 1
        else ", ".join(options[:-1]) + ", or " + options[-1]
    )
    raise ValueError(f"{problem} Use {hint}.")
