"""Define and inspect a Lance Dataset."""

from zephon.debug import DatasetInspector
from zephon.io import Dataset

# Discovery reads the dataset's newest manifest and pins to it, so dataset.path
# names that version: later appends do not change this run's shards.
dataset = Dataset.from_path("lance_demo", "data/lance_demo", fmt="lance")
print(dataset.path)

with DatasetInspector(dataset) as inspector:
    print(inspector.read(shard_id=0, sample_index=0))
