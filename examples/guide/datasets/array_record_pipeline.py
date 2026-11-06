"""Decode and print the records from array_record_write_shards.py."""

from typing import Any

import msgpack

from zephon import Pipeline
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource


def decode(record: Any) -> dict[str, Any]:
    """Decode the MessagePack bytes written by array_record_write_shards.py."""
    return msgpack.unpackb(record)


dataset = Dataset.from_path("array_record_demo", "data/array_record_demo")
work_source = StaticMixtureWorkSource(
    datasets=[dataset],
    mixture=MixtureSpec({"array_record_demo": 1.0}),
    shuffle_shards=False,
    exhausted_policy="redistribute",  # Read every record once, then stop.
)
pipeline = Pipeline(work_source).map_transform(decode).options(runner="inline")

for sample in pipeline:
    print(sample.payload)
