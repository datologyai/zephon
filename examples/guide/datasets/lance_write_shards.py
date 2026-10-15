"""Write a small Lance dataset, one fragment per shard."""

import lance
import pyarrow as pa

table = pa.table(
    {
        "id": pa.array(range(8), pa.int64()),
        "text": pa.array([f"record {i}" for i in range(8)]),
    }
)

# Each fragment becomes one Zephon shard, so max_rows_per_file sets the shard size.
lance.write_dataset(table, "data/lance_demo", max_rows_per_file=4)
