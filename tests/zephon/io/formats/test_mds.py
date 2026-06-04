import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streaming")
from streaming import MDSWriter

from tests._helpers import catalog_locators
from zephon.io.catalog import extra_codec
from zephon.io.dataset import Dataset
from zephon.io.formats import ensure_builtin_formats
from zephon.io.formats.base import get_format
from zephon.io.formats.mds import _MDSExtraCodec
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
    ensure_builtin_formats(required={"mds"})
    handler = get_format("mds")
    _, locators = catalog_locators(dataset)
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


def test_catalog_reverse_lookup_raw_and_zip(tmp_path):
    """The RESUME reverse lookup resolves raw and zip basenames to (slot, role)."""
    dataset_dir = tmp_path / "rev"
    dataset_dir.mkdir()
    columns = {"text": "str"}
    with MDSWriter(
        out=str(dataset_dir), columns=columns, compression="zstd:3", hashes=["sha1"]
    ) as writer:
        for i in range(10):
            writer.write({"text": f"s{i}"})

    dataset = Dataset.from_path(name="rev", path=str(dataset_dir))
    catalog, locators = catalog_locators(dataset)
    loc = locators[0]
    assert loc.zip is not None  # compressed -> has a zip companion
    assert catalog.reverse_lookup(loc.raw.basename) == (0, "raw")
    assert catalog.reverse_lookup(loc.zip.basename) == (0, "zip")
    assert catalog.reverse_lookup("does-not-exist.mds") is None


def _write_mds_dataset(dataset_dir: Path, **writer_kwargs) -> list[dict[str, object]]:
    """Write 10 small samples and return them (sorted by ``value``)."""
    dataset_dir.mkdir()
    columns = {"text": "str", "value": "int"}
    samples: list[dict[str, object]] = [
        {"text": f"{'x' * 512}{idx}", "value": idx} for idx in range(10)
    ]
    with MDSWriter(out=str(dataset_dir), columns=columns, **writer_kwargs) as writer:
        for sample in samples:
            writer.write(sample)
    return samples


def _read_all_via_catalog_locators(dataset: Dataset) -> list[dict[str, object]]:
    """Open every shard straight from catalog-synthesized locators and read it.

    Feeds ``locator_at`` output directly to ``open_shard``, so the codec-decoded
    ``extra`` — not the discovery-time mapping — is what the reader consumes.
    """
    ensure_builtin_formats(required={"mds"})
    handler = get_format("mds")
    _, locators = catalog_locators(dataset)
    assert dataset.path is not None
    resolver = DirectResolver(LocalFSBackend(root=Path(dataset.path)))

    observed: list[dict[str, object]] = []
    for shard_id, locator in locators.items():
        extra = dict(locator.extra) if locator.extra else {}
        template = extra["_streaming_template"]
        assert isinstance(template, dict)
        # The per-shard 'samples' must survive the codec's header hoist (it is
        # stripped at encode and re-attached from the num_rows column).
        assert template["samples"] == dataset.shard_index[shard_id]
        shard = handler.open_shard(locator, resolver.resolve(locator))
        try:
            assert len(shard) == dataset.shard_index[shard_id]
            for row_idx in range(len(shard)):
                row = shard[row_idx]
                observed.append({"text": str(row["text"]), "value": int(row["value"])})
        finally:
            shard.close()
    return sorted(observed, key=lambda item: item["value"])


@pytest.mark.parametrize(
    ("description", "writer_kwargs", "expected_min_shards"),
    [
        ("single", {}, 1),
        ("multi", {"size_limit": 2048}, 2),  # ragged tail -> per-shard samples
    ],
)
def test_catalog_extra_serves_reader(
    tmp_path, description, writer_kwargs, expected_min_shards
):
    """The mds codec's decoded ``extra`` feeds the real reader.

    The template minus ``samples`` is constant across shards, so both layouts
    take the header hoist; the multi-shard case proves each shard gets its own
    ``samples`` back (the tail shard is shorter).
    """
    dataset_dir = tmp_path / f"codec_{description}"
    samples = _write_mds_dataset(dataset_dir, **writer_kwargs)

    dataset = Dataset.from_path(name=f"codec_{description}", path=str(dataset_dir))
    assert len(dataset.shard_index) >= expected_min_shards
    assert _read_all_via_catalog_locators(dataset) == samples


def test_mds_extra_roundtrips_through_default_codec(tmp_path, monkeypatch):
    """With no mds codec registered, the generic rest path serves the reader.

    The whole ``extra`` (template included) rides the catch-all per-shard
    blob and the reader still decodes correctly.
    """
    monkeypatch.delitem(extra_codec._CODECS, "mds", raising=False)

    dataset_dir = tmp_path / "default_codec"
    samples = _write_mds_dataset(dataset_dir, size_limit=2048)

    dataset = Dataset.from_path(name="default_codec", path=str(dataset_dir))
    assert len(dataset.shard_index) >= 2
    assert _read_all_via_catalog_locators(dataset) == samples


def test_mds_codec_owns_nothing_when_templates_differ():
    """Templates that differ beyond ``samples`` flow through the generic rest path."""
    codec = _MDSExtraCodec()
    template = {"version": 2, "samples": 4, "size_limit": 1024}
    metas = [
        {"_streaming_template": dict(template)},
        {"_streaming_template": {**template, "size_limit": 2048}},
    ]
    encoded = codec.encode(metas, np.array([4, 4], dtype=np.int64))
    assert encoded.owned_keys == frozenset()
    assert encoded.header_blob is None
    assert codec.decode(0, None, {}, 4, encoded.flags) is None


def test_mds_codec_owns_nothing_when_template_lacks_samples():
    """A template without ``samples`` is not hoistable (decode would fabricate it)."""
    codec = _MDSExtraCodec()
    metas = [{"_streaming_template": {"version": 2, "size_limit": 1024}}]
    encoded = codec.encode(metas, np.array([4], dtype=np.int64))
    assert encoded.owned_keys == frozenset()
    assert encoded.header_blob is None


def test_mds_codec_raises_on_samples_count_contradiction():
    """``encode`` refuses to strip a ``samples`` that disagrees with the counts.

    Decode re-attaches ``samples`` from the catalog's ``num_rows`` column, so a
    template that contradicts its shard count cannot round-trip; it must fail
    at build time rather than be silently rewritten.
    """
    codec = _MDSExtraCodec()
    metas = [{"_streaming_template": {"version": 2, "samples": 5}}]
    with pytest.raises(ValueError, match="self-contradictory"):
        codec.encode(metas, np.array([7], dtype=np.int64))


def test_mds_codec_samples_check_is_unconditional():
    """A ``samples`` contradiction raises even after the header hoist is dead.

    Slot 1 breaks template constancy (the codec will own nothing), but slot 3's
    contradiction must still fail at build time rather than surface as a reader
    mismatch at ``open_shard``.
    """
    codec = _MDSExtraCodec()
    metas = [
        {"_streaming_template": {"version": 2, "samples": 3}},
        {"_streaming_template": {"version": 3, "samples": 4}},
        {"_streaming_template": {"version": 2, "samples": 5}},
        {"_streaming_template": {"version": 2, "samples": 99}},
    ]
    with pytest.raises(ValueError, match="slot 3"):
        codec.encode(metas, np.array([3, 4, 5, 7], dtype=np.int64))


def test_discover_counts_matches_discover(tmp_path):
    """``discover_counts`` agrees with ``discover`` on ``(shard_id, num_rows)``."""
    dataset_dir = tmp_path / "counts"
    _write_mds_dataset(dataset_dir, size_limit=2048)

    ensure_builtin_formats(required={"mds"})
    handler = get_format("mds")
    storage = LocalFSBackend(root=Path("/"))
    shard_index, _ = handler.discover(str(dataset_dir), storage)
    ids, counts = handler.discover_counts(str(dataset_dir), storage)

    assert len(ids) >= 2
    assert ids.tolist() == sorted(shard_index)
    assert counts.tolist() == [shard_index[sid] for sid in sorted(shard_index)]


def test_discover_counts_defers_to_discover_on_malformed_index(tmp_path):
    """An index entry without ``samples`` falls back to ``discover`` (which raises)."""
    dataset_dir = tmp_path / "broken"
    _write_mds_dataset(dataset_dir)
    index_path = dataset_dir / "index.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    del data["shards"][0]["samples"]
    index_path.write_text(json.dumps(data), encoding="utf-8")

    ensure_builtin_formats(required={"mds"})
    handler = get_format("mds")
    storage = LocalFSBackend(root=Path("/"))
    with pytest.raises(ValueError, match="missing 'samples'"):
        handler.discover_counts(str(dataset_dir), storage)


def test_mds_reader_handles_root_index_with_subdir_basenames(tmp_path):
    """Ensure we can read when the root index basenames include a subdirectory.

    Simulates a layout like:
      root/index.json  (basenames "1350/shard.00000.mds[.zstd]")
      root/1350/index.json
      root/1350/shard.00000.mds[.zstd]
    This mirrors datasets where shard basenames are relative to a split subdir.
    """
    dataset_root = tmp_path / "root_with_subdir_basenames"
    split_dir = dataset_root / "1350"
    split_dir.mkdir(parents=True)

    columns = {"text": "str", "value": "int", "payload": "str"}
    samples = [
        {
            "text": f"sample-{idx}",
            "value": idx,
            "payload": f"{'x' * 64}{idx}",
        }
        for idx in range(30)
    ]

    # Write an MDS shard inside the split subdirectory (with compression).
    with MDSWriter(
        out=str(split_dir), columns=columns, compression="zstd:3", hashes=["sha1"]
    ) as writer:
        for sample in samples:
            writer.write(sample)

    # Load the subdir index and construct a root-level index where basenames
    # include the subdirectory prefix (e.g., "1350/shard.00000.mds").
    sub_index_path = split_dir / "index.json"
    with sub_index_path.open("r", encoding="utf-8") as fh:
        sub_index = json.load(fh)

    shards = []
    for entry in sub_index.get("shards", []):
        entry = dict(entry)
        raw = dict(entry.get("raw_data", {}))
        zipd = dict(entry.get("zip_data", {})) if entry.get("zip_data") else None
        if "basename" in raw:
            raw["basename"] = f"1350/{raw['basename']}"
        entry["raw_data"] = raw
        if zipd is not None and "basename" in zipd:
            zipd["basename"] = f"1350/{zipd['basename']}"
        entry["zip_data"] = zipd
        shards.append(entry)

    root_index = {
        "version": sub_index.get("version", 2),
        "shards": shards,
    }
    dataset_root.mkdir(exist_ok=True)
    with (dataset_root / "index.json").open("w", encoding="utf-8") as fh:
        json.dump(root_index, fh)

    dataset = Dataset.from_path(
        name="root_with_subdir_basenames", path=str(dataset_root)
    )
    ensure_builtin_formats(required={"mds"})
    handler = get_format("mds")
    _, locators = catalog_locators(dataset)

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

    observed: list[dict[str, object]] = []
    for shard_id in locators.keys():
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
