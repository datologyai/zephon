"""Define and inspect an ArrayRecord Dataset."""

from zephon.debug import DatasetInspector
from zephon.io import Dataset

dataset = Dataset.from_path("array_record_demo", "data/array_record_demo")

# Each sample is the record's raw bytes, in the encoding chosen by the writer.
with DatasetInspector(dataset) as inspector:
    print(inspector.read(shard_id=0, sample_index=0))
