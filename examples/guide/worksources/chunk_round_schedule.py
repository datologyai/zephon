"""Drive the mixture from a schedule measured in chunk rounds."""

from zephon.io import Dataset
from zephon.work import MixtureSchedule, PhaseConstantSchedule, StaticMixtureWorkSource

code = Dataset.from_path("code", "s3://my-bucket/corpora/code")
web = Dataset.from_path("web", "s3://my-bucket/corpora/web")

# Breakpoints are (chunk_round, raw weight); each weight holds until the next.
schedule = MixtureSchedule(
    {
        "code": PhaseConstantSchedule([(0, 0.1), (1_000, 0.9)]),
        "web": PhaseConstantSchedule([(0, 0.9), (1_000, 0.1)]),
    }
)

work_source = StaticMixtureWorkSource(datasets=[code, web], mixture=schedule, seed=42)
