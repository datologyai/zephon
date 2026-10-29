# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import Any

import pytest

from zephon._internal.utils.tensor_utils import (
    count_valid_tokens,
    flatten_sequences,
    labels_to_loss_mask,
    mask_eos_labels,
    mask_padding_labels,
    pad_field,
    padding_lengths_to_mask,
    positions_from_segment_lengths,
    positions_to_cu_seqlens,
    positions_to_document_ids,
    sequence_positions,
    stack_sequences,
)


@pytest.mark.parametrize("framework", [None, "numpy", "torch"])
def test_document_membership_ignores_padding_and_keeps_inputs(
    framework: str | None,
) -> None:
    positions = [[0, 1, 0, 0], [0, 0, 1, 2]]
    padding = [[False, False, False, True], [False, False, False, False]]
    if framework == "numpy":
        np = pytest.importorskip("numpy")
        positions, padding = np.array(positions), np.array(padding)
    elif framework == "torch":
        torch = pytest.importorskip("torch")
        positions, padding = torch.tensor(positions), torch.tensor(padding)
    result = positions_to_document_ids(positions, padding, framework, flatten=True)
    values = result.tolist() if framework else result
    assert values == [[1, 1, 2, 0], [4, 5, 5, 5]]
    assert (positions.tolist() if framework else positions) == [
        [0, 1, 0, 0],
        [0, 0, 1, 2],
    ]


def test_document_membership_rejects_incompatible_list_shapes() -> None:
    for positions, padding in (
        ([[0]], []),
        ([[0, 1]], [[False]]),
        ([[[0]]], [[False]]),
    ):
        with pytest.raises(ValueError, match="scalar positions matching"):
            positions_to_document_ids(positions, padding, None, flatten=False)


@pytest.mark.parametrize("source", ["list", "numpy", "torch"])
@pytest.mark.parametrize("dtype_name", ["int32", "float32"])
def test_stack_sequences_numpy(source: str, dtype_name: str) -> None:
    np = pytest.importorskip("numpy")
    rows = [[1, 2, 3], [4, 5, 6]]
    if source == "numpy":
        sequences = [np.asarray(row) for row in rows]
    elif source == "torch":
        torch = pytest.importorskip("torch")
        sequences = [torch.tensor(row) for row in rows]
    else:
        sequences = rows

    dtype = getattr(np, dtype_name)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = stack_sequences(sequences, dtype, "numpy")
    assert out.tolist() == rows
    assert out.dtype == dtype


def test_stack_sequences_numpy_empty() -> None:
    np = pytest.importorskip("numpy")
    out = stack_sequences([], np.int32, "numpy")
    assert out.shape == (0,)
    assert out.dtype == np.int32


def test_shared_padding_preserves_tuple_and_checks_integer_range() -> None:
    np = pytest.importorskip("numpy")
    assert pad_field((1, 2), 2, 9) == (1, 2, 9, 9)
    for dtype, value in ((np.uint16, 128001), (np.int32, 2**31)):
        with pytest.raises(ValueError, match="not representable"):
            pad_field(np.array([1, 2], dtype=dtype), 1, value)


def test_sequence_padding_preserves_numpy_scalars() -> None:
    np = pytest.importorskip("numpy")
    value = np.int32(7)
    tail = np.array([0, 1], dtype=np.int32)
    for seq in ([value], (value,)):
        out = pad_field(seq, 2, 0, tail=tail)
        assert type(out) is type(seq)
        assert out[0] is value
        assert list(out) == [7, 0, 1]
        assert all(isinstance(item, np.int32) for item in out)


def test_padding_rejects_unsupported_containers() -> None:
    for seq in ({"tokens": [1, 2]}, "12", 12, object()):
        with pytest.raises(TypeError, match="unsupported field type.*expected"):
            pad_field(seq, 1, 0)


def test_list_padding_does_not_convert_to_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")

    def unexpected_array(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("list padding must not allocate a throwaway array")

    monkeypatch.setattr(np, "asarray", unexpected_array)
    assert pad_field([1, 2], 2, 9) == [1, 2, 9, 9]
    assert pad_field((1, 2), 2, 9, scalar=True) == (1, 2, 9, 9)
    with pytest.raises(ValueError, match="scalar token sequence"):
        pad_field([[1, 2]], 1, 0, scalar=True)


def test_padding_does_not_infer_torch_from_module_name() -> None:
    np = pytest.importorskip("numpy")

    class TorchNamedArray(np.ndarray):
        __module__ = "not_a_torch_tensor"

    seq = np.asarray([1, 2], dtype=np.int32).view(TorchNamedArray)
    out = pad_field(seq, 1, 9, scalar=True)
    assert out.tolist() == [1, 2, 9]
    assert out.dtype == np.int32


def test_padded_stack_uses_explicit_position_tails_without_mutating_inputs() -> None:
    np = pytest.importorskip("numpy")
    rows = [np.array([0, 1, 0], dtype=np.int32), np.array([0], dtype=np.int32)]
    out = stack_sequences(
        rows,
        np.int32,
        "numpy",
        length=5,
        tails=[positions_from_segment_lengths((5 - len(row),)) for row in rows],
    )
    assert out.tolist() == [[0, 1, 0, 0, 1], [0, 0, 1, 2, 3]]
    assert [row.tolist() for row in rows] == [[0, 1, 0], [0]]


def test_mask_padding_labels_list() -> None:
    out = mask_padding_labels([[1, 2, 3], [4, 5, 6]], [1, 2], -100, None)
    assert out == [[1, 2, -100], [4, -100, -100]]


def test_mask_padding_labels_numpy_does_not_mutate() -> None:
    import numpy as np

    labels = np.array([[1, 2, 3], [4, 5, 6]])
    out = mask_padding_labels(labels, [1, 2], -100, "numpy")
    assert out.tolist() == [[1, 2, -100], [4, -100, -100]]
    assert labels.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_mask_padding_labels_torch() -> None:
    torch = pytest.importorskip("torch")

    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = mask_padding_labels(labels, [1, 2], -100, "torch")
    assert out.tolist() == [[1, 2, -100], [4, -100, -100]]


def test_mask_padding_labels_zero_is_noop() -> None:
    out = mask_padding_labels([[1, 2, 3], [4, 5, 6]], [0, 0], -100, None)
    assert out == [[1, 2, 3], [4, 5, 6]]


def test_mask_padding_labels_ragged_rows() -> None:
    labels = [[], [4], [5, 6, 7, 8]]
    assert mask_padding_labels(labels, [0, 0, 2], -100, None) == [
        [],
        [4],
        [5, 6, -100, -100],
    ]
    assert labels == [[], [4], [5, 6, 7, 8]]


def test_mask_padding_labels_validates_each_ragged_row() -> None:
    with pytest.raises(AssertionError, match="pad_lengths"):
        mask_padding_labels([[1, 2, 3], [4]], [0, 2], -100, None)


@pytest.mark.parametrize("framework", [None, "numpy", "torch"])
@pytest.mark.parametrize("shifted", [False, True])
def test_padding_lengths_to_mask_without_input_padding(
    framework: str | None, shifted: bool
) -> None:
    module = pytest.importorskip(framework) if framework is not None else None
    tokens = stack_sequences([[0] * 5] * 2, module.int32 if module else None, framework)
    # A single trailing pad disappears from the input after next-token shifting.
    mask = padding_lengths_to_mask(
        tokens, [int(shifted)] * 2, framework, shifted=shifted
    )
    expected = [[False] * (5 - int(shifted))] * 2
    if framework is None:
        assert mask == expected
    else:
        assert mask.tolist() == expected
        assert str(mask.dtype) in ("bool", "torch.bool")


@pytest.mark.parametrize("framework", ["numpy", "torch"])
@pytest.mark.parametrize("shape", [(0, 5), (2, 0)])
def test_padding_lengths_to_mask_empty_dimensions(
    framework: str, shape: tuple[int, int]
) -> None:
    module = pytest.importorskip(framework)
    mask = padding_lengths_to_mask(
        module.zeros(shape), [0] * shape[0], framework, shifted=True
    )
    assert mask.shape == (shape[0], max(0, shape[1] - 1))
    assert str(mask.dtype) in ("bool", "torch.bool")


@pytest.mark.parametrize("framework", [None, "numpy", "torch"])
@pytest.mark.parametrize("shifted", [False, True])
def test_padding_lengths_to_mask(framework: str | None, shifted: bool) -> None:
    rows = [[0] * 5 for _ in range(4)]
    module = pytest.importorskip(framework) if framework is not None else None
    tokens = stack_sequences(rows, module.int32 if module else None, framework)
    out = padding_lengths_to_mask(tokens, [0, 1, 3, 5], framework, shifted=shifted)
    expected = [
        [False, False, False, False, False],
        [False, False, False, False, True],
        [False, False, True, True, True],
        [True, True, True, True, True],
    ]
    if shifted:
        expected = [row[:-1] for row in expected]
    if module is None:
        assert out == expected
        assert all(type(value) is bool for row in out for value in row)
        assert tokens == rows
    else:
        assert out.tolist() == expected
        assert str(out.dtype) in ("bool", "torch.bool")
        assert tokens.tolist() == rows
        if framework == "torch":
            assert out.device == tokens.device


@pytest.mark.parametrize("shifted", [False, True])
def test_padding_lengths_to_mask_ragged_rows(shifted: bool) -> None:
    out = padding_lengths_to_mask(
        [[], [0], [0, 0, 0]], [0, 1, 2], None, shifted=shifted
    )
    assert out == (
        [[], [], [False, True]] if shifted else [[], [True], [False, True, True]]
    )


@pytest.mark.parametrize("framework", [None, "numpy", "torch"])
@pytest.mark.parametrize("pad_lengths", [[-1, 0], [4, 0], [1], [0, 0, 0]])
def test_padding_lengths_to_mask_rejects_invalid_lengths(
    framework: str | None, pad_lengths: list[int]
) -> None:
    module = pytest.importorskip(framework) if framework is not None else None
    tokens = stack_sequences(
        [[1, 2, 3], [4, 5, 6]], module.int32 if module else None, framework
    )
    with pytest.raises(ValueError, match="padding_length"):
        padding_lengths_to_mask(tokens, pad_lengths, framework, shifted=False)


@pytest.mark.parametrize("framework", ["numpy", "torch"])
@pytest.mark.parametrize("shape", [(3,), (1, 2, 3)])
def test_padding_lengths_to_mask_requires_token_rows(
    framework: str, shape: tuple[int, ...]
) -> None:
    module = pytest.importorskip(framework)
    with pytest.raises(ValueError, match="2-D"):
        padding_lengths_to_mask(module.zeros(shape), [0], framework, shifted=False)


@pytest.mark.parametrize("framework", [None, "numpy", "torch"])
def test_flatten_sequences_preserves_row_order(framework: str | None) -> None:
    rows = [[1, 2, 3], [4, 5, 6]]
    if framework == "numpy":
        np = pytest.importorskip("numpy")
        sequences = np.array(rows, dtype=np.int32)[:, 1:]
    elif framework == "torch":
        torch = pytest.importorskip("torch")
        sequences = torch.tensor(rows, dtype=torch.int32)[:, 1:]
    else:
        sequences = [row[1:] for row in rows]

    # Shifted tensor/array rows are non-contiguous and must still flatten correctly.
    out = flatten_sequences(sequences, framework)
    if framework is None:
        assert out == [2, 3, 5, 6]
    else:
        assert out.tolist() == [2, 3, 5, 6]
        assert out.dtype == sequences.dtype


@pytest.mark.parametrize("framework", [None, "numpy", "torch"])
def test_count_valid_tokens_uses_ignore_index(framework: str | None) -> None:
    labels = [[2, -7, 3], [-7, -7, 6]]
    if framework == "numpy":
        labels = pytest.importorskip("numpy").array(labels)
    elif framework == "torch":
        labels = pytest.importorskip("torch").tensor(labels)

    count = count_valid_tokens(labels, -7, framework)
    assert count == 3
    assert type(count) is int
    assert count_valid_tokens(labels[:0], -7, framework) == 0


@pytest.mark.parametrize("framework", [None, "numpy", "torch"])
def test_labels_to_loss_mask_preserves_labels_and_uses_float32(
    framework: str | None,
) -> None:
    rows = [[2, -7, 3], [-7, -7, 6]]
    labels = rows
    if framework == "numpy":
        np = pytest.importorskip("numpy")
        labels = np.array([[0] + row for row in rows], dtype=np.int64)[:, 1:]
    elif framework == "torch":
        torch = pytest.importorskip("torch")
        labels = torch.tensor([[0] + row for row in rows], dtype=torch.long)[:, 1:]

    mask = labels_to_loss_mask(labels, -7, framework)
    expected = [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    if framework is None:
        assert mask == expected
        assert all(type(value) is float for row in mask for value in row)
    else:
        assert mask.tolist() == expected
        assert mask.dtype == pytest.importorskip(framework).float32
        if framework == "torch":
            assert mask.device == labels.device
    mask[0][0] = 0.0
    assert (labels if framework is None else labels.tolist()) == rows


@pytest.mark.parametrize("framework", [None, "numpy", "torch"])
@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize(
    ("positions", "boundaries", "maximum"),
    [
        ([], [0], 0),
        ([0], [0, 1], 1),
        ([0, 0, 0], [0, 1, 2, 3], 1),
        ([0, 1, 2], [0, 3], 3),
        ([0, 1, 0, 1, 2, 0], [0, 2, 5, 6], 3),
    ],
)
def test_positions_to_cu_seqlens_segment_boundaries(
    framework: str | None,
    flatten: bool,
    positions: list[int],
    boundaries: list[int],
    maximum: int,
) -> None:
    rows = [positions]
    if framework == "numpy":
        rows = pytest.importorskip("numpy").array(rows)
    elif framework == "torch":
        rows = pytest.importorskip("torch").tensor(rows)

    cu, maxima = positions_to_cu_seqlens(rows, framework, flatten=flatten)
    expected = boundaries if flatten else [boundaries]
    assert (cu if framework is None else cu.tolist()) == expected
    assert (maxima if flatten or framework is None else maxima.tolist()) == (
        maximum if flatten else [maximum]
    )
    if flatten:
        assert type(maxima) is int
    if framework is not None:
        module = pytest.importorskip(framework)
        assert cu.dtype == module.int32
        if not flatten:
            assert maxima.dtype == module.int32
            assert maxima.shape == (1,)
        if framework == "torch":
            assert cu.device == rows.device
            if not flatten:
                assert maxima.device == rows.device


@pytest.mark.parametrize("framework", [None, "numpy", "torch"])
def test_positions_to_cu_seqlens_padding_depends_on_segment_count(
    framework: str | None,
) -> None:
    rows = [list(range(8192)), list(range(4096)) * 2]
    if framework == "numpy":
        rows = pytest.importorskip("numpy").array(rows)
    elif framework == "torch":
        rows = pytest.importorskip("torch").tensor(rows)

    cu, maxima = positions_to_cu_seqlens(rows, framework, flatten=False)
    assert (cu if framework is None else cu.tolist()) == [
        [0, 8192, 8192],
        [0, 4096, 8192],
    ]
    assert (maxima if framework is None else maxima.tolist()) == [8192, 4096]


@pytest.mark.parametrize("framework", ["numpy", "torch"])
@pytest.mark.parametrize("flatten", [False, True])
def test_positions_to_cu_seqlens_zero_rows(framework: str, flatten: bool) -> None:
    module = pytest.importorskip(framework)
    rows = module.empty((0, 7), dtype=module.int64)
    cu, maxima = positions_to_cu_seqlens(rows, framework, flatten=flatten)
    assert cu.tolist() == ([0] if flatten else [])
    if flatten:
        assert maxima == 0
        assert type(maxima) is int
    else:
        assert cu.shape == (0, 1)
        assert maxima.shape == (0,)
        assert maxima.dtype == module.int32


@pytest.mark.parametrize("framework", [None, "numpy", "torch"])
@pytest.mark.parametrize("positions", [[[1, 2]], [[[0], [1]]]])
def test_positions_to_cu_seqlens_rejects_malformed_positions(
    framework: str | None,
    positions: list,
) -> None:
    rows = positions
    if framework == "numpy":
        rows = pytest.importorskip("numpy").array(rows)
    elif framework == "torch":
        rows = pytest.importorskip("torch").tensor(rows)
    with pytest.raises(ValueError, match="start at zero|2-D"):
        positions_to_cu_seqlens(rows, framework, flatten=False)


@pytest.mark.parametrize("framework", [None, "numpy", "torch"])
def test_mask_eos_labels_matches_inputs_without_mutating_sources(
    framework: str | None,
) -> None:
    inputs = [[1, 2, 7], [0, 2, 2]]
    labels = [[2, 7, 2], [2, 2, 8]]
    tokens: Any = inputs
    targets: Any = labels
    if framework == "numpy":
        np = pytest.importorskip("numpy")
        tokens, targets = (
            np.array(inputs, dtype=np.int32),
            np.array(labels, dtype=np.int32),
        )
    elif framework == "torch":
        torch = pytest.importorskip("torch")
        tokens, targets = (
            torch.tensor(inputs, dtype=torch.int32),
            torch.tensor(labels, dtype=torch.int32),
        )
    masked = mask_eos_labels(targets, tokens, 2, -9, framework)
    assert (masked.tolist() if framework else masked) == [[2, -9, 2], [2, -9, -9]]
    assert (targets.tolist() if framework else targets) == labels
    assert (tokens.tolist() if framework else tokens) == inputs
    if framework:
        assert masked.dtype == targets.dtype


@pytest.mark.parametrize("framework", [None, "numpy", "torch"])
def test_sequence_positions_returns_independent_contiguous_rows(
    framework: str | None,
) -> None:
    rows = [[11, 12, 13], [21, 22, 23]]
    tokens: Any = rows
    if framework == "numpy":
        np = pytest.importorskip("numpy")
        tokens = np.array(rows, dtype=np.int64)
    elif framework == "torch":
        torch = pytest.importorskip("torch")
        tokens = torch.tensor(rows, dtype=torch.int64)
    positions = sequence_positions(tokens, framework)
    assert (positions.tolist() if framework else positions) == [[0, 1, 2], [0, 1, 2]]
    positions[0][0] = 9
    assert positions[1][0] == 0
    assert (tokens.tolist() if framework else tokens) == rows
    if framework:
        assert positions.dtype == tokens.dtype
    if framework == "torch":
        assert positions.device == tokens.device
        assert positions.view(-1).tolist() == [9, 1, 2, 0, 1, 2]
    elif framework == "numpy":
        assert positions.flags.c_contiguous
