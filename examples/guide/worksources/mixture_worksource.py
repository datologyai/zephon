"""Combine two Datasets into one training mixture."""

from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

code = Dataset.from_path("code", "s3://my-bucket/corpora/code")
web = Dataset.from_path("web", "s3://my-bucket/corpora/web")

# Weights are relative, not absolute: {"code": 1, "web": 3} is the same mix.
work_source = StaticMixtureWorkSource(
    datasets=[code, web],
    mixture=MixtureSpec({"code": 0.25, "web": 0.75}),
    seed=42,
)
