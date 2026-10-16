"""Pin a Lance dataset to an earlier version by naming its manifest."""

from zephon.io import Dataset

# Any committed manifest under _versions/ is a valid dataset root. Passing one
# reads that version, which is what resuming a run against the same shards needs.
dataset = Dataset.from_path(
    "lance_demo", "data/lance_demo/_versions/18446744073709551614.manifest"
)
print(dataset.shard_count(), dataset.total())
