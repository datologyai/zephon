"""Pack processor-prepared text and images, then collate them for training."""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from zephon import Pipeline

# Example output of prepare_document.
document = {
    "input_ids": torch.tensor([1, 32000, 32000, 32000, 32000, 42, 43, 2]),
    "loss_mask": torch.tensor([0, 0, 0, 0, 0, 1, 1, 1]),
    "pixel_values": torch.zeros(1, 3, 224, 224),  # [images, C, H, W]
    "model_positions": torch.tensor(  # [3, tokens]
        [
            [0, 1, 1, 1, 1, 3, 4, 5],  # Temporal coordinate.
            [0, 1, 1, 2, 2, 3, 4, 5],  # Height coordinate.
            [0, 1, 2, 1, 2, 3, 4, 5],  # Width coordinate.
        ]
    ),
}


def merge_documents(documents: list[Any]) -> dict[str, Any]:
    """Combine the documents selected for one pack, keeping image order."""
    return {
        "input_ids": torch.cat([d["input_ids"] for d in documents]),
        "loss_mask": torch.cat([d["loss_mask"] for d in documents]),
        "positions": torch.cat([torch.arange(len(d["input_ids"])) for d in documents]),
        "pixel_values": torch.cat([d["pixel_values"] for d in documents]),
        "model_positions": torch.cat([d["model_positions"] for d in documents], dim=-1),
    }


def collate_images(
    values: Sequence[Any], *, length: int, shifted: bool
) -> torch.Tensor:
    """Combine all images in the batch along their image axis."""
    return torch.cat(values)


def collate_positions(
    values: Sequence[Any], *, length: int, shifted: bool
) -> torch.Tensor:
    """Pad token coordinates and stack them as [3, batch_size, input_length]."""
    padded = torch.stack(
        [F.pad(value, (0, length - value.shape[-1])) for value in values], dim=1
    )
    return padded[..., :-1] if shifted else padded


# prepare_document is your preprocessing function, with the output shown above.
pipeline = (
    Pipeline(work_source)
    .map_transform(prepare_document)
    .pack_sequences(
        max_length=9,
        algorithm="first_fit",
        num_bins=2,
        pack_payloads=merge_documents,
    )
    .batch(2, drop_last=False)
)

for sample_batch in pipeline:
    training_batch = sample_batch.to_training(
        dtype=torch.long,
        return_labels=True,
        pad_to_length=9,  # Eight model inputs after shifting.
        pad_token_id=0,
        field_collators={
            "pixel_values": collate_images,
            "model_positions": collate_positions,
        },
    )
