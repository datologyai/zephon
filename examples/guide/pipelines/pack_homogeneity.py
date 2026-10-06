"""Restrict which mixture components may share a packed sequence."""

from zephon import Pipeline
from zephon.io import Dataset
from zephon.ops import DomainGroups
from zephon.work import StaticMixtureWorkSource

mixture = {"web": 0.4, "books": 0.2, "math": 0.2, "code": 0.2}
datasets = [
    Dataset.from_path(name, f"s3://my-bucket/corpora/{name}") for name in mixture
]
work_source = StaticMixtureWorkSource(datasets, mixture, seed=42)

# homogeneity="full" would keep every sequence to a single component. Each
# group keeps its own bins and is flushed separately, so a flush can leave a
# partial sequence per group.
pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text")
    .pack_flat(
        max_length=4097,
        algorithm="first_fit",
        num_bins=8,
        pad_token_id=0,
        homogeneity="group",
        groups=DomainGroups({"prose": ["web", "books"], "formal": ["math", "code"]}),
    )
)
