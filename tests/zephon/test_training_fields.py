"""Conversion contracts for padded multimodal documents and exact target masks."""

import copy
from collections.abc import Sequence
from typing import Any

import pytest

from zephon.types import SampleBatch, SampleMeta, SampleRecord


def batch(*payloads: dict[str, Any], padding: int = 0) -> SampleBatch:
    return SampleBatch(
        records=tuple(
            SampleRecord(
                meta=SampleMeta(
                    sample_id=(0, 0, i),
                    lane_id=0,
                    chunk_id=0,
                    tags={"_packing_metadata": {"padding_length": padding}},
                ),
                payload=payload,
            )
            for i, payload in enumerate(payloads)
        )
    )


@pytest.fixture(params=[None, "numpy", "torch"])
def dtype(request: pytest.FixtureRequest) -> Any:
    return None if request.param is None else pytest.importorskip(request.param).int64


def values(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


@pytest.mark.parametrize("shift", [False, True])
def test_padding_is_positional_and_preserves_supervision(
    dtype: Any, shift: bool
) -> None:
    payloads = [
        {
            "input_ids": [2, 7, 2, 2, 8],
            "positions": [0, 1, 2, 0, 1],
            "loss_mask": [0, 1, 1, 0, 1],
        },
        {"input_ids": [2, 9, 2], "positions": [0, 1, 2], "loss_mask": [0, 1, 1]},
    ]
    original = copy.deepcopy(payloads)
    result = batch(*payloads).to_training(
        dtype=dtype,
        return_labels=shift,
        return_loss_mask=shift,
        return_num_valid_tokens=shift,
        pad_to_length=6,
        pad_token_id=2,
        return_padding_mask=True,
        return_cu_seqlens=True,
    )
    assert values(result["input_ids"]) == (
        [[2, 7, 2, 2, 8], [2, 9, 2, 2, 2]]
        if shift
        else [[2, 7, 2, 2, 8, 2], [2, 9, 2, 2, 2, 2]]
    )
    assert values(result["padding_mask"]) == (
        [[False] * 5, [False] * 3 + [True] * 2]
        if shift
        else [[False] * 5 + [True], [False] * 3 + [True] * 3]
    )
    assert values(result["cu_seqlens"]) == (
        [[0, 3, 5], [0, 3, 5]] if shift else [[0, 3, 5, 6], [0, 3, 6, 6]]
    )
    if shift:
        assert values(result["labels"]) == [
            [7, 2, -100, 8, -100],
            [9, 2, -100, -100, -100],
        ]
        assert result["num_valid_tokens"] == 5
    assert payloads == original


def test_added_padding_combines_with_existing_padding_metadata(dtype: Any) -> None:
    source = batch({"input_ids": [2, 8, 2, 2]}, padding=1)
    output = source.to_training(
        dtype=dtype,
        pad_to_length=6,
        pad_token_id=2,
        return_labels=True,
        return_padding_mask=True,
    )
    assert values(output["labels"]) == [[8, 2, -100, -100, -100]]
    assert values(output["padding_mask"]) == [[False, False, False, True, True]]
    assert source.records[0].meta.padding_length == 1


@pytest.mark.parametrize("tokens", [[], [2]])
def test_padding_empty_or_all_padding_document(dtype: Any, tokens: list[int]) -> None:
    output = batch({"input_ids": tokens}, padding=len(tokens)).to_training(
        dtype=dtype,
        pad_to_length=3,
        pad_token_id=2,
        return_labels=True,
        return_padding_mask=True,
        return_num_valid_tokens=True,
    )
    assert values(output["labels"]) == [[-100, -100]]
    assert values(output["padding_mask"]) == [[True, True]]
    assert output["num_valid_tokens"] == 0


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("mode", ["preserve", "sequence"])
def test_segment_mask_uses_target_boundaries_not_bos_eos_or_model_positions(
    dtype: Any, flatten: bool, mode: str
) -> None:
    source = batch(
        {
            "input_ids": [2, 11, 2, 2, 12, 2],
            "positions": [0, 1, 2, 0, 1, 2],
            "loss_mask": [1, 1, 0, 1, 1, 1],
        }
    )
    output = source.to_training(
        dtype=dtype,
        return_labels=True,
        mask_segment_starts=True,
        return_loss_mask=True,
        return_num_valid_tokens=True,
        return_cu_seqlens=True,
        position_mode=mode,
        flatten=flatten,
    )
    expected = [11, -100, -100, 12, 2]
    assert values(output["labels"]) == (expected if flatten else [expected])
    assert output["num_valid_tokens"] == 3
    assert values(output["loss_mask"]) == (
        [1.0, 0.0, 0.0, 1.0, 1.0] if flatten else [[1.0, 0.0, 0.0, 1.0, 1.0]]
    )
    assert values(output["cu_seqlens"]) == ([0, 3, 5] if flatten else [[0, 3, 5]])
    baseline = source.to_training(dtype=dtype, return_labels=True)
    assert values(baseline["labels"]) == [[11, -100, 2, 12, 2]]


def test_continuation_start_has_no_target_to_mask(dtype: Any) -> None:
    # A wrap fragment starts a new packed document at zero. Its first token is
    # absent from the target slice; the following content stays supervised.
    output = batch(
        {"input_ids": [31, 32, 33, 41], "positions": [0, 1, 2, 0]}
    ).to_training(
        dtype=dtype,
        return_labels=True,
        mask_segment_starts=True,
    )
    assert values(output["labels"]) == [[32, 33, -100]]


@pytest.mark.parametrize("positions", [[1, 2, 3], [[0, 1, 2]] * 3, [0, 1]])
def test_segment_mask_rejects_invalid_original_positions(
    dtype: Any, positions: Any
) -> None:
    with pytest.raises(ValueError, match="positions"):
        batch({"input_ids": [1, 2, 3], "positions": positions}).to_training(
            dtype=dtype,
            return_labels=True,
            mask_segment_starts=True,
        )


def test_segment_mask_requires_boundaries() -> None:
    with pytest.raises(ValueError, match="requires return_labels"):
        batch({"input_ids": [1, 2]}).to_training(mask_segment_starts=True)
    with pytest.raises(ValueError, match="positions"):
        batch({"input_ids": [1, 2]}).to_training(
            return_labels=True, mask_segment_starts=True
        )


@pytest.mark.parametrize("shift", [False, True])
@pytest.mark.parametrize("flatten", [False, True])
def test_field_collators_own_media_and_mrope_layout(shift: bool, flatten: bool) -> None:
    torch = pytest.importorskip("torch")
    payloads = [
        {
            "input_ids": [1, 2, 3],
            "positions": [0, 1, 2],
            "pixels": torch.ones((2, 3), dtype=torch.float32),
            "model_positions": torch.tensor([[0, 1, 2], [0, 4, 4], [0, 5, 5]]),
        },
        {
            "input_ids": [4, 5],
            "positions": [0, 1],
            "pixels": torch.ones((1, 3), dtype=torch.float32) * 2,
            "model_positions": torch.tensor([[0, 1], [0, 6], [0, 7]]),
        },
    ]
    snapshots = [(p["pixels"].clone(), p["model_positions"].clone()) for p in payloads]
    contexts = []

    def pixels(items: Sequence[Any], length: int, return_labels: bool) -> Any:
        contexts.append((length, return_labels))
        return torch.cat(tuple(items), dim=0)

    def mrope(items: Sequence[Any], length: int, return_labels: bool) -> Any:
        contexts.append((length, return_labels))
        padded = [
            torch.nn.functional.pad(item, (0, length - item.shape[-1]))
            for item in items
        ]
        result = torch.stack(padded, dim=1)
        return result[..., :-1] if return_labels else result

    result = batch(*payloads).to_training(
        dtype=torch.long,
        return_labels=shift,
        pad_to_length=5,
        pad_token_id=0,
        field_collators={"pixels": pixels, "model_positions": mrope},
        rename_fields={"pixels": "pixel_values"},
        flatten=flatten,
        return_cu_seqlens=True,
    )
    assert contexts == [(5, shift), (5, shift)]
    assert result["pixel_values"].shape == (3, 3)
    assert result["pixel_values"].dtype == torch.float32
    assert result["model_positions"].shape == (3, 2, 5 - int(shift))
    assert result["model_positions"][2, 1, 1] == 7
    for payload, (original_pixels, original_positions) in zip(payloads, snapshots):
        assert torch.equal(payload["pixels"], original_pixels)
        assert torch.equal(payload["model_positions"], original_positions)


def test_callbacks_empty_exclusion_missing_field_and_reserved_names() -> None:
    calls = []

    def collect(items: Sequence[Any], length: int, shift: bool) -> Any:
        calls.append(items)
        return list(items)

    assert batch().to_training(field_collators={"media": collect}) == {
        "ids": [],
        "texts": [],
        "media": [],
    }
    assert calls == []
    result = batch({"input_ids": [1, 2], "media": "image"}).to_training(
        dtype=None, field_collators={"media": collect}, exclude_fields=("media",)
    )
    assert "media" not in result
    with pytest.raises(ValueError, match="media"):
        batch({"input_ids": [1, 2]}).to_training(field_collators={"media": collect})
    with pytest.raises(ValueError, match="reserved"):
        batch().to_training(field_collators={"positions": collect})
    with pytest.raises(TypeError, match="callable"):
        batch().to_training(field_collators={"media": 1})
    with pytest.raises(ValueError, match="equal token lengths"):
        batch(
            {"input_ids": [1], "media": []}, {"input_ids": [2, 3], "media": []}
        ).to_training(dtype=None, field_collators={"media": collect})


@pytest.mark.parametrize(
    "options",
    [
        {"pad_to_length": -1, "pad_token_id": 0},
        {"pad_to_length": True, "pad_token_id": 0},
        {"pad_to_length": 3},
        {"pad_token_id": 0},
        {"pad_to_length": 3, "pad_token_id": -1},
        {"pad_to_length": 3, "pad_token_id": True},
    ],
)
def test_invalid_padding_options(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="pad_"):
        batch().to_training(**options)


def test_padding_rejects_truncation_and_non_token_extras() -> None:
    with pytest.raises(ValueError, match="truncation"):
        batch({"input_ids": [1, 2, 3]}).to_training(pad_to_length=2, pad_token_id=0)
    with pytest.raises(ValueError, match="field_collators"):
        batch({"input_ids": [1, 2], "pixels": [[1, 2], [3, 4]]}).to_training(
            pad_to_length=3, pad_token_id=0, extra_fields=("pixels",)
        )
    with pytest.raises(ValueError, match="padding_length"):
        batch({"input_ids": [1, 2]}, padding=3).to_training(
            pad_to_length=3, pad_token_id=0
        )


@pytest.mark.parametrize("framework", ["numpy", "torch"])
def test_padding_preserves_array_inputs_without_mutation(framework: str) -> None:
    module = pytest.importorskip(framework)
    convert = module.asarray if framework == "numpy" else module.tensor
    tokens = convert([2, 6, 2], dtype=module.int64)
    positions = convert([0, 1, 2], dtype=module.int64)
    masks = convert([0, 1, 1], dtype=module.int64)
    output = batch(
        {"input_ids": tokens, "positions": positions, "loss_mask": masks}
    ).to_training(
        dtype=module.int64,
        pad_to_length=5,
        pad_token_id=2,
        return_labels=True,
        return_padding_mask=True,
        return_cu_seqlens=True,
    )
    assert values(output["labels"]) == [[6, 2, -100, -100]]
    assert values(output["padding_mask"]) == [[False, False, False, True]]
    assert values(tokens) == [2, 6, 2]
    assert values(positions) == [0, 1, 2]
    assert values(masks) == [0, 1, 1]
    if framework == "torch":
        assert output["input_ids"].device == tokens.device
