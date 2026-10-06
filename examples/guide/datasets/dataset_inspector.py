"""Read raw samples from a Dataset without building a Pipeline."""

from zephon.debug import DatasetInspector
from zephon.io import Dataset, InMemoryShard

dataset = Dataset.from_dict(
    "demo",
    {0: InMemoryShard([{"text": "hello"}, {"text": "world"}])},
)

# A context manager, so the shards it opens are closed promptly.
with DatasetInspector(dataset) as inspector:
    print(inspector.read(shard_id=0, sample_index=1))
