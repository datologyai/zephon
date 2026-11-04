"""Define and inspect an ArrayRecord Dataset, then decode its records."""

from typing import Any

import msgpack

from zephon.debug import DatasetInspector
from zephon.io import Dataset

dataset = Dataset.from_path("array_record_demo", "data/array_record_demo")

# Each sample is the record's raw bytes: {"data": b"..."}.
with DatasetInspector(dataset) as inspector:
    print(inspector.read(shard_id=0, sample_index=0))


def decode(payload: Any) -> dict[str, Any]:
    """Turn a record's bytes back into the dictionary that was written."""
    return msgpack.unpackb(payload["data"])


# In a pipeline, decode with a map_transform right after the work source:
#     Pipeline(work_source).map_transform(decode, parallelism=4)
