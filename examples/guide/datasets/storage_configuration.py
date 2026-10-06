"""Give the shard cache a fast local directory and a size limit."""

from zephon import Pipeline
from zephon.io import CacheOptions, Dataset, StoreOptions
from zephon.work import MixtureSpec, StaticMixtureWorkSource

dataset = Dataset.from_path("wikipedia", "s3://my-bucket/corpora/wikipedia")
work_source = StaticMixtureWorkSource([dataset], MixtureSpec({"wikipedia": 1.0}))

# The limit has to fit the shard being prepared, compressed and decompressed.
# All ranks on a node share this directory, so size it for the whole node.
pipeline = Pipeline(work_source).options(
    io_options=StoreOptions(
        cache=CacheOptions(root="/mnt/nvme/zephon-cache", limit_bytes=512 << 30),
    )
)
