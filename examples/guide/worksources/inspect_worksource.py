"""Ask a work source how many samples it will produce."""

from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

code = Dataset.from_path("code", "s3://my-bucket/corpora/code")
web = Dataset.from_path("web", "s3://my-bucket/corpora/web")

work_source = StaticMixtureWorkSource(
    datasets=[code, web],
    mixture=MixtureSpec({"code": 0.25, "web": 0.75}),
    seed=42,
)

# Infinite when the configuration never stops the source on its own.
print(f"{work_source.total_samples} samples before stopping")
