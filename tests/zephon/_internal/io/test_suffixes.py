import pytest

from zephon._internal.io.suffixes import (
    AUTO_DETECT_ORDER,
    JSONL_SUFFIXES,
    REMOTE_SCAN_FORMATS,
    detect_format,
    format_of,
)
from zephon._internal.utils.compression import COMPRESSION_SUFFIXES


def test_jsonl_suffixes_cover_every_compression() -> None:
    assert set(JSONL_SUFFIXES) == {
        ".jsonl",
        *(f".jsonl{suffix}" for suffix in COMPRESSION_SUFFIXES),
    }


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["a.jsonl"], "jsonl"),
        (["a.jsonl.zst", "b.jsonl.gz"], "jsonl"),
        (["a.vortex"], "vortex"),
        (["a.parquet"], "parquet"),
        (["a.parquet", "b.jsonl.gz"], "jsonl"),
        (["a.parquet", "b.vortex"], "vortex"),
        (["a.json", "b.json.gz", "index.json"], None),
        ([], None),
    ],
)
def test_detect_format(names: list[str], expected: str | None) -> None:
    assert detect_format(names) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("data%2Ftrain.jsonl.zst", "jsonl"),
        ("0000.parquet", "parquet"),
        ("a.vortex", "vortex"),
        ("index.json", None),
        ("c4-train.00000.json.gz", None),
    ],
)
def test_format_of(name: str, expected: str | None) -> None:
    assert format_of(name) == expected


def test_remote_scan_formats_are_detectable() -> None:
    assert REMOTE_SCAN_FORMATS <= {kind for kind, _ in AUTO_DETECT_ORDER}
    assert "vortex" not in REMOTE_SCAN_FORMATS
