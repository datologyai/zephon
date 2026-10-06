"""Checkpoint with one Pipeline per data-parallel group."""

from torch.distributed.device_mesh import init_device_mesh

from zephon import Pipeline

# Only the ranks that load data build a Pipeline -- the others are broadcast
# the batch -- so the coordination group is the data-parallel one, and the
# same two numbers serve as both identities.
mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
dp_mesh = mesh["dp"]

pipeline = Pipeline(work_source).options(
    dp_degree=dp_mesh.size(),
    dp_group_id=dp_mesh.get_local_rank(),
    world_size=dp_mesh.size(),
    global_rank=dp_mesh.get_local_rank(),
    aggregate_dir="s3://my-bucket/runs/aggregate",
    run_id="pretrain-7b-2026-03-14",
)
