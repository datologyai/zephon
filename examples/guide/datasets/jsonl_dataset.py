"""Define and inspect a JSONL Dataset."""

from zephon.debug import DatasetInspector
from zephon.io import Dataset

# Without an index.json alongside the shards, Zephon reads every line once to
# find where each sample starts.
dataset = Dataset.from_path("conversations", "data/conversations", fmt="jsonl")

with DatasetInspector(dataset) as inspector:
    print(inspector.read(shard_id=0, sample_index=0))
