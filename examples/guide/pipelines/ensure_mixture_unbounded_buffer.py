"""Hold the mixture by dropping oversupplied records instead of buffering."""

from zephon import Pipeline
from zephon.io import Dataset
from zephon.work import StaticMixtureWorkSource

code = Dataset.from_path("code", "s3://my-bucket/corpora/code")
web = Dataset.from_path("web", "s3://my-bucket/corpora/web")
work_source = StaticMixtureWorkSource([code, web], {"code": 0.25, "web": 0.75}, seed=42)

# max_buffer_size=None drops the excess from oversupplied components at each
# flush: closer to target, at the cost of lost data and more memory.
pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text")
    .ensure_mixture(max_buffer_size=None)
    .batch(microbatch_size=32)
)
