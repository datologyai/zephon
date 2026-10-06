"""Define and inspect a litData Dataset."""

from zephon.debug import DatasetInspector
from zephon.io import Dataset

# fmt is optional: Zephon infers the format from the index.json beside the
# chunks. Keep the two together when you move the dataset.
dataset = Dataset.from_path("litdata_demo", "data/litdata_demo", fmt="litdata")

with DatasetInspector(dataset) as inspector:
    print(inspector.read(shard_id=0, sample_index=0))
