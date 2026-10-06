"""The same batch, shaped for two training frameworks."""

from typing import Any

from zephon import SampleBatch


def for_torchtitan(batch: SampleBatch) -> dict[str, Any]:
    """One flat sequence per microbatch, with the count the loss normalizes by."""
    return batch.to_training(
        return_labels=True,
        flatten=True,
        return_padding_mask=True,
        return_num_valid_tokens=True,
    )


def for_megatron(batch: SampleBatch) -> dict[str, Any]:
    """[microbatch_size, sequence_length] tensors, a float loss mask, cu_seqlens."""
    return batch.to_training(
        return_labels=True,
        return_loss_mask=True,
        return_cu_seqlens=True,
        rename_fields={"input_ids": "tokens"},
    )
