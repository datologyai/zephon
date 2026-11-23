"""Sharding a Zephon pipeline across JAX processes."""

import jax
import numpy as np

from zephon import Pipeline

pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text", parallelism=8)
    .pack_flat(max_length=2049, algorithm="wrap")
    .batch(microbatch_size=32)
    .options(
        # ds[process_index::process_count] becomes this host's data identity.
        canonical_replicas=16,
        dp_degree=jax.process_count(),
        dp_group_id=jax.process_index(),
        world_size=jax.process_count(),
        global_rank=jax.process_index(),
        aggregate_dir="gs://my-bucket/runs/aggregate",
        run_id="pretrain-1b-2026-11-09",
        mtp_mode=True,
    )
)

for sample_batch in pipeline:
    # NumPy arrays instead of torch tensors; segment IDs as Grain's packers emit them.
    batch = sample_batch.to_training(
        dtype=np.int32,
        return_labels=True,
        return_document_ids=True,
    )
    state = train_step(state, jax.device_put(batch))
