# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from zephon._internal.io.resolvers.cache.layout import (
    validate_raw_cache_dataset_names,
)


def test_raw_cache_allows_nested_dataset_names(tmp_path: Path) -> None:
    validate_raw_cache_dataset_names(tmp_path, ["org/dataset"])


@pytest.mark.parametrize(
    "name",
    ["../escape", ".parquet-rg-cache", ".parquet-rg-cache/nested"],
)
def test_raw_cache_rejects_unsafe_or_reserved_dataset_names(
    tmp_path: Path,
    name: str,
) -> None:
    with pytest.raises(ValueError, match="unsafe inside cache root"):
        validate_raw_cache_dataset_names(tmp_path, [name])
