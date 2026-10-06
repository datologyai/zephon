"""Take the data identity from the data-parallel sub-mesh."""

from torch.distributed.device_mesh import init_device_mesh

from zephon import Pipeline

# 8 GPUs with tensor parallelism of 2 leaves 4 data-parallel groups. The TP,
# PP and CP degrees never reach Zephon: they do not change who reads what.
mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
dp_mesh = mesh["dp"]

pipeline = Pipeline(work_source).options(
    dp_degree=dp_mesh.size(),
    dp_group_id=dp_mesh.get_local_rank(),
)
