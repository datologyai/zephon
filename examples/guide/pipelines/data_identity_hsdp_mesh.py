"""Flatten replicate and shard dimensions into one data-parallel degree."""

from torch.distributed.device_mesh import init_device_mesh

from zephon import Pipeline

# Hybrid sharding spreads data parallelism over two mesh dimensions, but both
# need different data per rank, so Zephon is given their product.
mesh = init_device_mesh(
    "cuda", (2, 2, 2), mesh_dim_names=("dp_replicate", "dp_shard", "tp")
)
replicate, shard = mesh["dp_replicate"], mesh["dp_shard"]

pipeline = Pipeline(work_source).options(
    dp_degree=replicate.size() * shard.size(),
    dp_group_id=replicate.get_local_rank() * shard.size() + shard.get_local_rank(),
)
