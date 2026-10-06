"""Give each Dataset its own repetition behaviour."""

from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

code = Dataset.from_path("code", "s3://my-bucket/corpora/code")
web = Dataset.from_path("web", "s3://my-bucket/corpora/web")

# The small code set may be seen up to three times over; the large web set is
# read once and stops the source when it runs out.
work_source = StaticMixtureWorkSource(
    datasets=[code, web],
    mixture=MixtureSpec({"code": 0.25, "web": 0.75}),
    seed=42,
    exhausted_policy={"code": "repeat", "web": "stop"},
    max_repeats={"code": 2, "web": None},
    reshuffle_on_repeat={"code": True, "web": False},
)
