"""Build an image classification batch from prepared images and class labels."""

import torch

from zephon import Pipeline

pipeline = Pipeline(work_source).batch(microbatch_size=32)

for sample_batch in pipeline:
    records = sample_batch.records
    images = torch.stack([torch.as_tensor(r.payload["image"]) for r in records])
    labels = torch.as_tensor([r.payload["label"] for r in records])
