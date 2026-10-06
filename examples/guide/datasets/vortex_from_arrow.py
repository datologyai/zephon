"""Convert an Arrow table into a Vortex shard."""

import pyarrow as pa
import vortex as vx

table = pa.table({"text": ["first", "second"], "label": [0, 1]})

vx.io.write(vx.array(table), "data/vortex_demo/shard0.vortex")
