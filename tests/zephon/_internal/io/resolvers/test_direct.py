import sys
from pathlib import Path

import pytest

from zephon._internal.io.resolvers.direct import DirectResolver
from zephon._internal.io.storage.local import LocalFSBackend
from zephon._internal.io.types import ShardFile, ShardLocator


def _mk_locator(
    root: str,
    *,
    basename: str,
    bytes: int,
    zip_meta=None,
    compression=None,
    hashes=None,
):
    return ShardLocator(
        dataset="d",
        shard_id=0,
        format="jsonl",
        root=root,
        raw=ShardFile(basename=basename, bytes=bytes, hashes=hashes or {}),
        zip=zip_meta,
        compression=compression,
        extra=None,
    )


def test_direct_resolver_basic(tmp_path: Path) -> None:
    backend = LocalFSBackend(root=Path("/"))
    resolver = DirectResolver(backend)
    data = b"hello"
    raw = tmp_path / "raw.bin"
    raw.write_bytes(data)

    loc = _mk_locator(str(tmp_path), basename="raw.bin", bytes=len(data))
    ref = resolver.resolve(loc)
    assert ref.raw.path == raw
    assert ref.raw.bytes == len(data)
    resolver.touch(loc)  # no-op


def test_direct_resolver_truncated_raises(tmp_path: Path) -> None:
    backend = LocalFSBackend(root=Path("/"))
    resolver = DirectResolver(backend)
    raw = tmp_path / "raw.bin"
    raw.write_bytes(b"x")
    loc = _mk_locator(str(tmp_path), basename="raw.bin", bytes=10)
    with pytest.raises(ValueError):
        _ = resolver.resolve(loc)


def test_direct_resolver_hash_validation(tmp_path: Path) -> None:
    backend = LocalFSBackend(root=Path("/"))
    raw = tmp_path / "raw.bin"
    content = b"abc"
    raw.write_bytes(content)

    # md5 of 'abc' is known
    loc_ok = _mk_locator(
        str(tmp_path),
        basename="raw.bin",
        bytes=len(content),
        hashes={"md5": "900150983cd24fb0d6963f7d28e17f72"},
    )
    resolver_ok = DirectResolver(backend, validate_hash="md5")
    assert resolver_ok.resolve(loc_ok).raw.path.exists()

    loc_bad = _mk_locator(
        str(tmp_path),
        basename="raw.bin",
        bytes=len(content),
        hashes={"md5": "deadbeef"},
    )
    resolver_bad = DirectResolver(backend, validate_hash="md5")
    with pytest.raises(ValueError):
        _ = resolver_bad.resolve(loc_bad)


def test_direct_resolver_decompresses_zstd_when_missing_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = LocalFSBackend(root=Path("/"))
    resolver = DirectResolver(backend)

    zip_path = tmp_path / "shard.zst"
    zip_path.write_bytes(b"compressed")
    raw_path = tmp_path / "shard.raw"

    class _Zstd:
        @staticmethod
        def decompress(data):
            # Our stub just returns verbatim, good enough for the test
            return data

    monkeypatch.setitem(sys.modules, "zstd", _Zstd)

    zip_meta = ShardFile(basename="shard.zst", bytes=zip_path.stat().st_size, hashes={})
    loc = _mk_locator(
        str(tmp_path),
        basename="shard.raw",
        bytes=zip_path.stat().st_size,
        zip_meta=zip_meta,
        compression="zstd",
    )
    assert not raw_path.exists()
    ref = resolver.resolve(loc)
    # Decompression produced raw file and returned ref
    assert ref.raw.path.exists()
    assert ref.raw.path.read_bytes() == zip_path.read_bytes()


def test_direct_resolver_zip_missing_raises(tmp_path: Path) -> None:
    backend = LocalFSBackend(root=Path("/"))
    resolver = DirectResolver(backend)
    zip_meta = ShardFile(basename="missing.zst", bytes=1, hashes={})
    loc = _mk_locator(
        str(tmp_path),
        basename="raw.bin",
        bytes=1,
        zip_meta=zip_meta,
        compression="zstd",
    )
    with pytest.raises(FileNotFoundError):
        _ = resolver.resolve(loc)


def test_direct_resolver_unsupported_compression(tmp_path: Path) -> None:
    backend = LocalFSBackend(root=Path("/"))
    resolver = DirectResolver(backend)
    zipf = tmp_path / "archive.gz"
    zipf.write_bytes(b"gz")
    zip_meta = ShardFile(basename="archive.gz", bytes=zipf.stat().st_size, hashes={})
    loc = _mk_locator(
        str(tmp_path),
        basename="raw.bin",
        bytes=zipf.stat().st_size,
        zip_meta=zip_meta,
        compression="gzip",
    )
    with pytest.raises(RuntimeError):
        _ = resolver.resolve(loc)


def test_direct_resolver_reprepares_empty_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = LocalFSBackend(root=Path("/"))
    resolver = DirectResolver(backend)

    raw_path = tmp_path / "shard.raw"
    raw_path.write_bytes(b"")  # existing but empty
    zip_path = tmp_path / "shard.zst"
    payload = b"payload"
    zip_path.write_bytes(payload)

    attempts = {"count": 0}

    class _Zstd:
        @staticmethod
        def decompress(data):
            attempts["count"] += 1
            return data

    monkeypatch.setitem(sys.modules, "zstd", _Zstd)

    zip_meta = ShardFile(basename="shard.zst", bytes=len(payload), hashes={})
    loc = _mk_locator(
        str(tmp_path),
        basename="shard.raw",
        bytes=len(payload),
        zip_meta=zip_meta,
        compression="zstd",
    )
    ref = resolver.resolve(loc)
    assert ref.raw.path.read_bytes() == payload
    assert attempts["count"] == 1  # only decompressed once


def test_direct_resolver_retries_until_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = LocalFSBackend(root=Path("/"))
    resolver = DirectResolver(backend)

    zip_path = tmp_path / "shard.zst"
    payload = b"payload"
    zip_path.write_bytes(payload)
    raw_path = tmp_path / "shard.raw"

    attempts = {"count": 0}

    class _Zstd:
        @staticmethod
        def decompress(data):
            attempts["count"] += 1
            if attempts["count"] < 3:
                # simulate a broken decompression that returns nothing
                return b""
            return data

    monkeypatch.setitem(sys.modules, "zstd", _Zstd)

    zip_meta = ShardFile(basename="shard.zst", bytes=len(payload), hashes={})
    loc = _mk_locator(
        str(tmp_path),
        basename="shard.raw",
        bytes=len(payload),
        zip_meta=zip_meta,
        compression="zstd",
    )

    ref = resolver.resolve(loc)
    assert ref.raw.path.read_bytes() == payload
    assert attempts["count"] == 3


def test_direct_resolver_retry_exhaustion_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = LocalFSBackend(root=Path("/"))
    resolver = DirectResolver(backend)

    zip_path = tmp_path / "shard.zst"
    payload = b"payload"
    zip_path.write_bytes(payload)
    raw_path = tmp_path / "shard.raw"

    attempts = {"count": 0}

    class _Zstd:
        @staticmethod
        def decompress(data):
            attempts["count"] += 1
            # Always return nothing, triggering validation failure
            return b""

    monkeypatch.setitem(sys.modules, "zstd", _Zstd)

    zip_meta = ShardFile(basename="shard.zst", bytes=len(payload), hashes={})
    loc = _mk_locator(
        str(tmp_path),
        basename="shard.raw",
        bytes=len(payload),
        zip_meta=zip_meta,
        compression="zstd",
    )

    with pytest.raises(ValueError) as excinfo:
        _ = resolver.resolve(loc)
    assert "Raw shard empty" in str(excinfo.value)
    assert attempts["count"] == 3
