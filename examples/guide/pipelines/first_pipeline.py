"""Fetch and batch records from a StaticMixtureWorkSource."""

from zephon import Pipeline
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

code = Dataset.from_path("code", "s3://my-bucket/corpora/code")
web = Dataset.from_path("web", "s3://my-bucket/corpora/web")

work_source = StaticMixtureWorkSource(
    datasets=[code, web],
    mixture=MixtureSpec({"code": 0.25, "web": 0.75}),
    seed=42,
)

# Fetch is implicit, so this pipeline only groups the records it reads -- the
# right shape when the samples were prepared offline.
pipeline = Pipeline(work_source).batch(microbatch_size=32)

for sample_batch in pipeline:
    print(len(sample_batch.records), "records")
