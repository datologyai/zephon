"""Exchange checkpoint state through torchrun's TCPStore."""

import torch.distributed as dist

from zephon import Pipeline

# Launched with torchrun, whose agent runs a TCPStore at MASTER_ADDR and
# MASTER_PORT. Initializing the process group first also starts that store
# on rank 0 in setups where the agent doesn't share it.
dist.init_process_group("nccl")

pipeline = Pipeline(work_source).options(
    world_size=dist.get_world_size(),
    global_rank=dist.get_rank(),
    aggregate_store="torch://",
    run_id="pretrain-7b-2026-03-14",
)
