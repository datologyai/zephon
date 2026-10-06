"""Write a small MDS dataset with the streaming library."""

import numpy as np
from streaming import MDSWriter

# One encoding per field; every record written must match what it declares.
columns = {"text": "str", "tokens": "ndarray"}

with MDSWriter(out="data/mds_demo", columns=columns, compression="zstd") as writer:
    for i in range(1000):
        writer.write(
            {"text": f"sample {i}", "tokens": np.arange(i % 32 + 1, dtype=np.int32)}
        )
