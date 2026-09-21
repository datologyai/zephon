# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from zephon._internal.utils.tensor_utils import (
    count_valid_tokens,
    flatten_sequences,
    mask_padding_labels,
)


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
