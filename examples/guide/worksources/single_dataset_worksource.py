"""Build the simplest work source: one Dataset, one weight."""

from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

dataset = Dataset.from_path("wikipedia", "s3://my-bucket/corpora/wikipedia")

# seed is what makes the curriculum reproducible across runs; chunk_size
# defaults to 16,384 pointers per work chunk.
work_source = StaticMixtureWorkSource(
    datasets=[dataset],
    mixture=MixtureSpec({"wikipedia": 1.0}),
    seed=42,
)
