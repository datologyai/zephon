"""Fix the number of lanes so the run can be resumed on another topology."""

from torch.distributed.device_mesh import init_device_mesh

from zephon import Pipeline

mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
dp_mesh = mesh["dp"]

# Eight lanes over four groups: two lanes each. It cannot be changed later, so
# pick it for the largest dp_degree the run may reach, and keep every dp_degree
# the run uses a divisor of it.
pipeline = Pipeline(work_source).options(
    canonical_replicas=8,
    dp_degree=dp_mesh.size(),
    dp_group_id=dp_mesh.get_local_rank(),
)
