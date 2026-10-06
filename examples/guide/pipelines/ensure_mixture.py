"""Reorder records so the output tracks the work source's mixture."""

from zephon import Pipeline
from zephon.io import Dataset
from zephon.work import StaticMixtureWorkSource

code = Dataset.from_path("code", "s3://my-bucket/corpora/code")
web = Dataset.from_path("web", "s3://my-bucket/corpora/web")
work_source = StaticMixtureWorkSource([code, web], {"code": 0.25, "web": 0.75}, seed=42)

# Placed after tokenize, because the default weighting counts tokens.
# warn_tolerance logs when a component drifts more than 5 points off target.
pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text")
    .ensure_mixture(max_buffer_size=1000, warn_tolerance=0.05)
    .batch(microbatch_size=32)
)
