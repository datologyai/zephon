# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

import pytest

from zephon._internal.utils.tensor_utils import (
    count_valid_tokens,
    flatten_sequences,
    labels_to_loss_mask,
    mask_padding_labels,
    positions_to_cu_seqlens,
    stack_sequences,
)


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
