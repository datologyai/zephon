"""Resume on four data-parallel groups a run that started on eight."""

from torch.distributed.device_mesh import init_device_mesh

from zephon import Pipeline

mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
dp_mesh = mesh["dp"]

# The topology is new; canonical_replicas is not. Zephon refuses a checkpoint
# whose lane count changed, and hands each of the four groups two lanes.
pipeline = (
    Pipeline(work_source)
    .batch(microbatch_size=32)
    .options(
        canonical_replicas=8,
        dp_degree=dp_mesh.size(),
        dp_group_id=dp_mesh.get_local_rank(),
        world_size=dp_mesh.size(),
        global_rank=dp_mesh.get_local_rank(),
        aggregate_dir="s3://my-bucket/runs/aggregate",
        run_id="pretrain-7b-2026-03-14",
    )
)

pipeline.restore(checkpoint["data"])
