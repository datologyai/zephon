"""Name the Pipelines that checkpoint together."""

import torch

from zephon import Pipeline

# Every rank builds a Pipeline, so every rank checkpoints. Rank 0 leads and
# writes the combined checkpoint; the others hand it their progress through
# aggregate_dir, which run_id keeps apart from other jobs sharing it.
pipeline = Pipeline(work_source).options(
    world_size=torch.distributed.get_world_size(),
    global_rank=torch.distributed.get_rank(),
    aggregate_dir="s3://my-bucket/runs/aggregate",
    run_id="pretrain-7b-2026-03-14",
)
