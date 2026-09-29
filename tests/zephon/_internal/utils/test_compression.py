import bz2
import gzip
import lzma
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from zephon._internal.utils.compression import (
    COMPRESSION_SUFFIXES,
    compression_for_name,
    decompress_file,
    normalize_compression,
    open_decompressed,
    zstd,
)

_WRITERS = {"gzip": gzip.open, "bz2": bz2.open, "xz": lzma.open, "zstd": zstd.open}
# Large enough that each of the two frames exceeds zstd's 128 KiB block size.
_PAYLOAD = b"".join(b'{"i": %d}\n' % i for i in range(50_000))


def _write_streamed(path: Path, payload: bytes, compression: str) -> None:
    """Write two independent frames/members, as streaming compressors emit."""
    half = len(payload) // 2
    with _WRITERS[compression](path, "wb") as fh:
        fh.write(payload[:half])
    with _WRITERS[compression](path, "ab") as fh:
        fh.write(payload[half:])


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("gz", "gzip"),
        ("GZIP", "gzip"),
        ("bzip2", "bz2"),
        ("lzma", "xz"),
        ("zst", "zstd"),
        ("zstd:7", "zstd"),
        ("zstandard", "zstd"),
        ("lz4", "lz4"),
    ],
)
def test_normalize_compression_aliases(label: str, expected: str) -> None:
    assert normalize_compression(label) == expected


def test_normalize_compression_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported compression: br"):
        normalize_compression("br")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.jsonl.gz", "gzip"),
        ("a.jsonl.zst", "zstd"),
        ("a.jsonl.bz2", "bz2"),
        ("a.jsonl.xz", "xz"),
        ("a.jsonl", None),
        ("a.gzip", None),
    ],
)
def test_compression_for_name(name: str, expected: str | None) -> None:
    assert compression_for_name(name) == expected


def test_every_suffix_algorithm_is_decodable() -> None:
    assert set(COMPRESSION_SUFFIXES.values()) == set(_WRITERS)


@pytest.mark.parametrize("compression", sorted(_WRITERS))
def test_open_decompressed_path_reads_every_frame(
    tmp_path: Path, compression: str
) -> None:
    path = tmp_path / "data.cmp"
    _write_streamed(path, _PAYLOAD, compression)
    with open_decompressed(path, compression) as stream:
        assert stream.read() == _PAYLOAD


@pytest.mark.parametrize("compression", sorted(_WRITERS))
def test_open_decompressed_leaves_file_object_open(
    tmp_path: Path, compression: str
) -> None:
    path = tmp_path / "data.cmp"
    _write_streamed(path, _PAYLOAD, compression)
    with path.open("rb") as raw:
        with open_decompressed(raw, compression) as stream:
            assert stream.read() == _PAYLOAD
        assert not raw.closed


def test_decompress_file_handles_unsized_multi_frame_zstd(tmp_path: Path) -> None:
    """Streamed zstd output carries no content size and may span several frames."""
    src = tmp_path / "data.zst"
    _write_streamed(src, _PAYLOAD, "zstd")
    descriptor = src.read_bytes()[4]
    # Frame_Content_Size_flag == 0 and Single_Segment_flag == 0: no size field.
    assert descriptor & 0xE0 == 0

    dst = tmp_path / "data"
    decompress_file(src, dst, "zstd")
    assert dst.read_bytes() == _PAYLOAD


def test_open_decompressed_lz4(tmp_path: Path) -> None:
    lz4frame = pytest.importorskip("lz4.frame")
    path = tmp_path / "data.lz4"
    with lz4frame.open(path, "wb") as fh:
        fh.write(_PAYLOAD)
    with open_decompressed(path, "lz4") as stream:
        assert stream.read() == _PAYLOAD


@pytest.mark.skipif(
    sys.version_info < (3, 14), reason="compression.zstd is the codec from 3.14"
)
def test_zephon_imports_without_a_zstd_codec(tmp_path: Path) -> None:
    """compression.zstd is optional in CPython builds; other codecs keep working."""
    gz = tmp_path / "a.jsonl.gz"
    gz.write_bytes(gzip.compress(b'{"i": 1}\n'))
    # A fresh interpreter, so blocking the codec cannot leak into other tests.
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class _NoZstd(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name in ("_zstd", "compression.zstd"):
                    raise ImportError(f"blocked {name}")
                return None

        sys.meta_path.insert(0, _NoZstd())

        from zephon.io import Dataset
        from zephon._internal.checkpoint._codec import AggregationCodec
        from zephon._internal.utils.compression import open_decompressed

        with open_decompressed(sys.argv[1], "gzip") as stream:
            assert stream.read() == b'{"i": 1}\\n'
        for use_zstd in (
            lambda: open_decompressed(sys.argv[1], "zstd"),
            lambda: AggregationCodec().encode({"k": 1}),
        ):
            try:
                use_zstd()
            except RuntimeError as exc:
                assert "compression.zstd" in str(exc), exc
            else:
                raise AssertionError("zstd must fail without a codec module")
        print(Dataset.from_path("gz", sys.argv[2]).total())
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(gz), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"
