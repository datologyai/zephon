from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from zephon.io.dataset import Dataset
from zephon.io.formats import ensure_builtin_formats
from zephon.io.formats.base import get_format
from zephon.io.formats.litdata_support import LitDataWriter
from zephon.io.resolvers import DirectResolver
from zephon.io.storage import LocalFSBackend


@pytest.mark.parametrize("loader_kind", ["pytree", "tokens"])
@pytest.mark.parametrize("compression", [False, True])
def test_litdata_reader_handles_dataset(
    tmp_path: Path, loader_kind: str, compression: bool
) -> None:
    dataset_dir = (
        tmp_path / f"litdata_{loader_kind}_{'compressed' if compression else 'plain'}"
    )
    kwargs: dict[str, object] = {"loader": loader_kind}
    if loader_kind == "tokens":
        block_size = 4
        kwargs["block_size"] = block_size
        samples = [
            np.arange(i * block_size, (i + 1) * block_size, dtype=np.int32)
            for i in range(4)
        ]
    else:
        samples = [(idx, f"sample-{idx}") for idx in range(4)]
    if compression:
        kwargs["compression"] = "zstd:3"

    with LitDataWriter(out=str(dataset_dir), **kwargs) as writer:
        for sample in samples:
            writer.write(sample)

    dataset = Dataset.from_path(name="lit", path=str(dataset_dir))
    ensure_builtin_formats()
    handler = get_format("litdata")
    locators = handler.build_locators(dataset)
    assert dataset.path is not None
    resolver = DirectResolver(LocalFSBackend(root=Path(dataset.path)))

    observed: list[tuple[int, str]] = []
    for shard_id, locator in locators.items():
        local_ref = resolver.resolve(locator)
        chunk_meta: dict[str, object] = {}
        if isinstance(locator.extra, Mapping):
            candidate = locator.extra.get("chunk")
            if isinstance(candidate, Mapping):
                chunk_meta = dict(candidate)
        if compression:
            assert chunk_meta.get("compression") == "zstd:3"
            assert chunk_meta.get("chunk_zip_path")
        else:
            assert chunk_meta.get("compression") is None
        shard = handler.open_shard(locator, local_ref)
        try:
            assert len(shard) == dataset.shard_index[shard_id]
            for row_idx in range(len(shard)):
                row = shard[row_idx]
                if loader_kind == "tokens":
                    assert isinstance(row, np.ndarray)
                    observed.append(np.array(row))
                else:
                    observed.append((int(row[0]), str(row[1])))
        finally:
            shard.close()

    if loader_kind == "tokens":
        for expected, actual in zip(samples, observed, strict=True):
            assert isinstance(actual, np.ndarray)
            assert np.array_equal(actual, expected)
    else:
        expected = [(idx, f"sample-{idx}") for idx in range(len(samples))]
        assert observed == expected
