"""Tests for the public single-process DatasetInspector."""

from __future__ import annotations

from pathlib import Path

import pytest

from zephon._internal.io.resolvers import CacheManager
from zephon.build_index import build_index
from zephon.debug import DatasetInspector
from zephon.io import Dataset, InMemoryShard
from zephon.io.options import CacheOptions, StoreOptions


def test_inspector_reads_in_memory_payloads_and_preserves_order() -> None:
    dataset = Dataset.from_dict(
        "tiny",
        {0: InMemoryShard([{"value": 0}, {"value": 1}, {"value": 2}])},
    )

    with DatasetInspector(dataset) as inspector:
        assert inspector.read(0, 1) == {"value": 1}
        assert inspector.read_many(0, [2, 0, 2]) == [
            {"value": 2},
            {"value": 0},
            {"value": 2},
        ]
        assert inspector.read_many(0, []) == []


def test_inspector_reads_jsonl_payloads(tmp_path: Path) -> None:
    (tmp_path / "part-000.jsonl").write_text(
        '{"text": "first"}\n{"text": "second"}\n', encoding="utf-8"
    )
    dataset = Dataset.from_path("tiny", str(tmp_path), fmt="jsonl")

    with DatasetInspector(dataset) as inspector:
        assert inspector.read(0, 1) == {"text": "second"}


def test_inspector_rejects_reads_after_close() -> None:
    dataset = Dataset.from_dict("tiny", {0: InMemoryShard([{"value": 0}])})
    inspector = DatasetInspector(dataset)
    inspector.close()

    with pytest.raises(RuntimeError, match="DatasetInspector is closed"):
        inspector.read(0, 0)


def test_inspector_closes_cache_manager_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "part-000.jsonl").write_text(
        '{"text": "first"}\n', encoding="utf-8"
    )
    dataset = Dataset.from_path("tiny", str(dataset_root), fmt="jsonl")
    options = StoreOptions(cache=CacheOptions(enabled=True, root=tmp_path / "cache"))

    close_calls: list[CacheManager] = []
    original_close = CacheManager.close

    def tracking_close(manager: CacheManager) -> None:
        close_calls.append(manager)
        original_close(manager)

    monkeypatch.setattr(CacheManager, "close", tracking_close)

    with DatasetInspector(dataset, io_options=options) as inspector:
        assert inspector.read(0, 0) == {"text": "first"}
    inspector.close()

    assert len(close_calls) == 1


@pytest.mark.parametrize("with_index", [False, True])
def test_inspector_reads_parquet_with_and_without_index(
    tmp_path: Path, with_index: bool
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")

    root = tmp_path / "parquet"
    root.mkdir()
    table = pyarrow.table({"id": [0, 1, 2], "text": ["zero", "one", "two"]})
    parquet.write_table(table, root / "part-000.parquet", row_group_size=2)
    if with_index:
        build_index("parquet", root, progress=False)

    dataset = Dataset.from_path("tiny", str(root), fmt="parquet")
    with DatasetInspector(dataset) as inspector:
        rows = inspector.read_many(0, [2, 0, 2])

    assert [row["id"] for row in rows] == [2, 0, 2]
    assert [row["text"] for row in rows] == ["two", "zero", "two"]


def test_inspector_reads_litdata(tmp_path: Path) -> None:
    pytest.importorskip("litdata")
    from litdata.streaming.writer import BinaryWriter

    root = tmp_path / "litdata"
    writer = BinaryWriter(cache_dir=str(root), chunk_size=3)
    for index, text in enumerate(["zero", "one", "two"]):
        writer.add_item(index, {"id": index, "text": text})
    writer.done()
    writer.merge()

    dataset = Dataset.from_path("tiny", str(root), fmt="litdata")
    with DatasetInspector(dataset) as inspector:
        rows = inspector.read_many(0, [2, 0, 2])

    assert [row["id"] for row in rows] == [2, 0, 2]
    assert [row["text"] for row in rows] == ["two", "zero", "two"]


def test_inspector_reads_mds(tmp_path: Path) -> None:
    pytest.importorskip("streaming")
    from streaming import MDSWriter

    root = tmp_path / "mds"
    root.mkdir()
    with MDSWriter(out=str(root), columns={"id": "int", "text": "str"}) as writer:
        for index, text in enumerate(["zero", "one", "two"]):
            writer.write({"id": index, "text": text})

    dataset = Dataset.from_path("tiny", str(root), fmt="mds")
    with DatasetInspector(dataset) as inspector:
        rows = inspector.read_many(0, [2, 0, 2])

    assert [row["id"] for row in rows] == [2, 0, 2]
    assert [row["text"] for row in rows] == ["two", "zero", "two"]


@pytest.mark.integration
@pytest.mark.parametrize("with_index", [False, True])
def test_inspector_reads_vortex_with_and_without_index(
    tmp_path: Path, with_index: bool
) -> None:
    pytest.importorskip("vortex.io")
    import vortex

    root = tmp_path / "vortex"
    root.mkdir()
    vortex.io.write(
        vortex.array(
            [
                {"id": 0, "text": "zero"},
                {"id": 1, "text": "one"},
                {"id": 2, "text": "two"},
            ]
        ),
        str(root / "part-000.vortex"),
    )
    if with_index:
        build_index("vortex", root, progress=False)

    dataset = Dataset.from_path("tiny", str(root), fmt="vortex")
    with DatasetInspector(dataset) as inspector:
        rows = inspector.read_many(0, [2, 0, 2])

    assert [row["id"] for row in rows] == [2, 0, 2]
    assert [row["text"] for row in rows] == ["two", "zero", "two"]
