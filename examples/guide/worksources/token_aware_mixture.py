"""Mix by tokens rather than by samples."""

from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource, TokenEstimation

web = Dataset.from_path("web", "s3://my-bucket/corpora/web")
code = Dataset.from_path("code", "s3://my-bucket/corpora/code")

# Long web documents would otherwise contribute far more than half the tokens.
# "measure" tokenizes a small sample of each dataset to estimate its rate.
work_source = StaticMixtureWorkSource(
    datasets=[web, code],
    mixture=MixtureSpec({"web": 0.5, "code": 0.5}),
    seed=42,
    token_estimation=TokenEstimation(primer="measure", calibration_samples=2048),
)
