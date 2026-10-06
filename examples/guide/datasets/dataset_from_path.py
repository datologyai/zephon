"""Point a Dataset at a directory or prefix of shards."""

from zephon.io import Dataset

# The name is what a mixture refers to this dataset by later.
wikipedia = Dataset.from_path("wikipedia", "s3://my-bucket/corpora/wikipedia")

print(f"{wikipedia.shard_count()} shards, {wikipedia.total()} samples")
