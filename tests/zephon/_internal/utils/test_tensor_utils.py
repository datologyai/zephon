# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from zephon._internal.utils.tensor_utils import mask_padding_labels


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
