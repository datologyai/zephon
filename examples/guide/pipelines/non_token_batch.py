"""Build tensors yourself when the records are not token sequences."""

import torch

from zephon import Pipeline

pipeline = Pipeline(work_source).batch(microbatch_size=32)

for sample_batch in pipeline:
    # to_training only speaks token sequences, so read the payloads directly.
    records = sample_batch.records
    images = torch.stack([torch.as_tensor(r.payload["image"]) for r in records])
    labels = torch.as_tensor([r.payload["label"] for r in records])
