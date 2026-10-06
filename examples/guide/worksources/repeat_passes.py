"""Keep drawing samples until every dataset has completed three passes."""

from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

code = Dataset.from_path("code", "s3://my-bucket/corpora/code")
web = Dataset.from_path("web", "s3://my-bucket/corpora/web")

# Datasets that get there first keep repeating to hold the mixture while the
# others catch up, reshuffled on each pass unless reshuffle_on_repeat is off.
work_source = StaticMixtureWorkSource(
    datasets=[code, web],
    mixture=MixtureSpec({"code": 0.25, "web": 0.75}),
    seed=42,
    stop_after_passes=3,
)
