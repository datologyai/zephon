"""Write records into Vortex shards."""

import vortex as vx

records = [{"text": f"sample {i}", "label": i % 2} for i in range(1000)]

# One file per shard; 64-256 MB each is the range we recommend.
for shard, start in enumerate(range(0, len(records), 250)):
    array = vx.array(records[start : start + 250])
    vx.io.write(array, f"data/vortex_demo/shard{shard}.vortex")
