"""Give a Pipeline the topology that Streaming read from the environment."""

import torch

from zephon import Pipeline

# num_canonical_nodes=8 becomes eight lanes. Zephon does not read RANK or
# WORLD_SIZE itself, so the topology is passed in explicitly.
pipeline = (
    Pipeline(work_source)
    .tokenize(tokenizer_id="gpt2", field="text")
    .pack_flat(max_length=2049, algorithm="wrap", pad_token_id=0)
    .batch(microbatch_size=32)
    .options(
        canonical_replicas=8,
        world_size=torch.distributed.get_world_size(),
        global_rank=torch.distributed.get_rank(),
        aggregate_store="torch://",
        run_id="pretrain-1b-2026-11-09",
        mtp_mode=True,
    )
)
