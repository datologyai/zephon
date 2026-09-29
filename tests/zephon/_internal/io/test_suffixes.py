import pytest

from zephon._internal.io.suffixes import JSONL_SUFFIXES, detect_format
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
