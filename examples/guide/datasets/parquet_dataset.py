"""Define and inspect a Parquet Dataset."""

from zephon.debug import DatasetInspector
from zephon.io import Dataset

# Parquet carries its row count in each file footer, so a Dataset can be
# discovered without an index.json -- an index just makes it faster.
dataset = Dataset.from_path("wikipedia", "s3://my-bucket/corpora/wikipedia")

with DatasetInspector(dataset) as inspector:
    print(inspector.read(shard_id=0, sample_index=0))
