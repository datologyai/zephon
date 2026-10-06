"""Define and inspect an MDS Dataset."""

from zephon.debug import DatasetInspector
from zephon.io import Dataset

dataset = Dataset.from_path("mds_demo", "data/mds_demo", fmt="mds")

with DatasetInspector(dataset) as inspector:
    print(inspector.read(shard_id=0, sample_index=0))
