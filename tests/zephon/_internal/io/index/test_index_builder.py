from pathlib import Path
from threading import Event, Lock

import pytest

from tests.helpers.storage import MemoryStorage
from zephon._internal.io.index.index_builder import IndexBuilder, ShardInfo


class _JsonlLikeBuilder(IndexBuilder):
    suffixes = (".jsonl", ".jsonl.gz")

    def extract_shard_info(self, path: str, file_size: int) -> ShardInfo:
        return ShardInfo(basename=Path(path).name, bytes=file_size, num_rows=1)


def test_scan_directory_matches_suffixes(tmp_path: Path) -> None:
    for name in ("b.jsonl.gz", "a.jsonl", "c.json", "index.json", "d.parquet"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "e.jsonl").mkdir()  # directories never count as shards

    assert _JsonlLikeBuilder().scan_directory(tmp_path) == ["a.jsonl", "b.jsonl.gz"]


def test_scan_directory_without_matches_names_every_pattern(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"No \*\.jsonl, \*\.jsonl\.gz files found"):
        _JsonlLikeBuilder().scan_directory(tmp_path)


def test_parallel_indexing_is_bounded_and_keeps_progress_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Slow reads keep stable IDs and progress while respecting the configured bound."""
    storage = MemoryStorage(
        "s3://bucket/data", {f"{name}.jsonl": b"{}" for name in "abcdef"}
    )
    second_finished = Event()
    lock = Lock()
    active = peak = 0

    class OrderedBuilder(IndexBuilder):
        suffixes = (".jsonl",)

        def extract_shard_info(self, path: str, file_size: int) -> ShardInfo:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if path.endswith("a.jsonl"):
                    assert second_finished.wait(timeout=5)
                if path.endswith("b.jsonl"):
                    second_finished.set()
                return ShardInfo(path.rsplit("/", 1)[-1], file_size, 1)
            finally:
                with lock:
                    active -= 1

    index = OrderedBuilder(storage, max_workers=2).build(
        storage.root, progress_interval=2
    )
    assert peak == 2
    assert [shard["basename"] for shard in index["shards"]] == [
        f"{name}.jsonl" for name in "abcdef"
    ]
    output = capsys.readouterr().out
    assert "Processed 2/6 files" in output
    assert "Processed 4/6 files" in output
    assert "Processed 6/6 files" in output


def test_failed_read_preserves_existing_index(monkeypatch: pytest.MonkeyPatch) -> None:
    storage = MemoryStorage(
        "custom://data",
        {"a.jsonl": b"{}", "b.jsonl": b"{}", "index.json": b"existing index"},
    )
    builder = _JsonlLikeBuilder(storage)
    extract = builder.extract_shard_info

    def fail_on_second(path: str, file_size: int) -> ShardInfo:
        if path.endswith("b.jsonl"):
            raise PermissionError("backend denied access")
        return extract(path, file_size)

    monkeypatch.setattr(builder, "extract_shard_info", fail_on_second)
    with pytest.raises(ValueError, match="Failed to index shard custom://data/b.jsonl"):
        builder.create_index(storage.root, progress=False)
    assert storage.files[f"{storage.root}/index.json"] == b"existing index"


def test_sequential_indexing_and_invalid_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = MemoryStorage("custom://data", {"b.jsonl": b"{}", "a.jsonl": b"{}"})
    builder = _JsonlLikeBuilder(storage, max_workers=1)
    paths: list[str] = []

    def record_read(path: str, file_size: int) -> ShardInfo:
        paths.append(path)
        return ShardInfo(Path(path).name, file_size, 1)

    monkeypatch.setattr(builder, "extract_shard_info", record_read)
    builder.build(storage.root, progress=False)
    assert paths == [f"{storage.root}/a.jsonl", f"{storage.root}/b.jsonl"]
    with pytest.raises(ValueError, match="max_workers"):
        _JsonlLikeBuilder(storage, max_workers=0)
    with pytest.raises(ValueError, match="progress_interval"):
        builder.build(storage.root, progress_interval=0)
