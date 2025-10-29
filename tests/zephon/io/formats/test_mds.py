from pathlib import Path

import pytest

pytest.importorskip("streaming")
from streaming import MDSWriter

from zephon.io.dataset import Dataset
from zephon.io.formats import ensure_builtin_formats
from zephon.io.formats.base import get_format
from zephon.io.resolvers import DirectResolver
from zephon.io.storage import LocalFSBackend
from zephon.io.stores import FileBackedDatasetShardView


@pytest.mark.parametrize(
    ("description", "writer_kwargs", "expected_min_shards", "expect_zip"),
    [
        ("plain", {}, 1, False),
        ("compressed", {"compression": "zstd:3", "hashes": ["sha1"]}, 1, True),
        ("multi_shard", {"size_limit": 2048}, 2, False),
    ],
)
def test_mds_reader_handles_streaming_variants(
    tmp_path, description, writer_kwargs, expected_min_shards, expect_zip
):
    dataset_dir = tmp_path / f"streaming_{description}"
    dataset_dir.mkdir()

    columns = {"text": "str", "value": "int", "payload": "str"}
    samples = [
        {
            "text": f"sample-{idx}",
            "value": idx,
            "payload": f"{'x' * 1024}{idx}",
        }
        for idx in range(10)
    ]

    with MDSWriter(out=str(dataset_dir), columns=columns, **writer_kwargs) as writer:
        for sample in samples:
            writer.write(sample)

    dataset = Dataset.from_path(name=f"streaming_{description}", path=str(dataset_dir))
    ensure_builtin_formats()
    handler = get_format("mds")
    locators = handler.build_locators(dataset)
    assert dataset.path is not None
    resolver = DirectResolver(LocalFSBackend(root=Path(dataset.path)))
    view = FileBackedDatasetShardView(
        dataset=dataset,
        handler=handler,
        resolver=resolver,
        retry_attempts=1,
        retry_initial_backoff=0.0,
        retry_max_backoff=0.0,
    )

    assert len(locators) >= expected_min_shards

    observed: list[dict[str, object]] = []
    for shard_id, locator in locators.items():
        if expect_zip:
            assert locator.zip is not None
        else:
            assert locator.zip is None
        if "hashes" in writer_kwargs:
            hashes_present = set(locator.raw.hashes)
            if locator.zip is not None:
                hashes_present.update(locator.zip.hashes)
            for algo in writer_kwargs["hashes"]:
                assert algo in hashes_present

        shard, reused = view.open(shard_id)
        assert reused in (True, False)
        try:
            assert len(shard) == dataset.shard_index[shard_id]
            for row_idx in range(len(shard)):
                got = shard[row_idx]
                assert isinstance(got, tuple)
                row = got[0]
                observed.append(
                    {
                        "text": str(row["text"]),
                        "value": int(row["value"]),
                        "payload": str(row["payload"]),
                    }
                )
        finally:
            shard.close()

    expected = sorted(samples, key=lambda item: item["value"])
    assert sorted(observed, key=lambda item: item["value"]) == expected
