from pathlib import Path

import pytest

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
