"""Change the mixture at predetermined optimizer steps."""

from zephon import Pipeline
from zephon.io import Dataset
from zephon.work import MixturePhase, StaticMixtureWorkSource, StepMixturePlan

code = Dataset.from_path("code", "s3://my-bucket/corpora/code")
web = Dataset.from_path("web", "s3://my-bucket/corpora/web")

# Mostly web for the first 8,192 steps, then mostly code.
plan = StepMixturePlan(
    phases=[
        MixturePhase(step=0, proportions={"code": 0.2, "web": 0.8}),
        MixturePhase(step=8_192, proportions={"code": 0.8, "web": 0.2}),
    ],
    total_steps=80_000,
)

work_source = StaticMixtureWorkSource(
    datasets=[code, web], mixture=plan, chunk_size=128, seed=42
)
# Records per optimizer step across all data-parallel ranks.
pipeline = (
    Pipeline(work_source).batch(4).options(canonical_replicas=8, global_batch_size=32)
)
