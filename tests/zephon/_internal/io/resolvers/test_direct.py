import bz2
import gzip
import lzma
from pathlib import Path

import pytest

import zephon._internal.io.resolvers.direct as direct_mod
from zephon._internal.io.resolvers.direct import DirectResolver
from zephon._internal.io.storage.local import LocalFSBackend
from zephon._internal.io.types import RemoteShardRef, ShardFile, ShardLocator
from zephon._internal.utils.compression import zstd


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


def _write_compressed(path: Path, payload: bytes, compression: str) -> None:
    """Write ``payload`` as a streaming (multi-frame, size-less) compressed file."""
    openers = {"gzip": gzip.open, "bz2": bz2.open, "xz": lzma.open, "zstd": zstd.open}
    half = len(payload) // 2
    with openers[compression](path, "wb") as fh:
        fh.write(payload[:half])
    with openers[compression](path, "ab") as fh:  # second frame / member
        fh.write(payload[half:])


@pytest.mark.parametrize("compression", ["gzip", "bz2", "xz", "zstd"])
def test_direct_resolver_decompresses_when_missing_raw(
    tmp_path: Path, compression: str
) -> None:
    backend = LocalFSBackend(root=Path("/"))
    resolver = DirectResolver(backend)

    payload = b"row\n" * 10_000
    zip_path = tmp_path / "shard.cmp"
    _write_compressed(zip_path, payload, compression)
    raw_path = tmp_path / "shard.raw"

    zip_meta = ShardFile(basename="shard.cmp", bytes=zip_path.stat().st_size, hashes={})
    loc = _mk_locator(
        str(tmp_path),
        basename="shard.raw",
        bytes=len(payload),
        zip_meta=zip_meta,
        compression=compression,
    )
    assert not raw_path.exists()
    ref = resolver.resolve(loc)
    assert ref.raw.path == raw_path
    assert raw_path.read_bytes() == payload
    # The temp sibling is gone once the raw file is published.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shard.cmp", "shard.raw"]


def test_direct_resolver_decodes_long_shard_names(tmp_path: Path) -> None:
    """The temp file must not add to a name already near NAME_MAX (255 bytes)."""
    resolver = DirectResolver(LocalFSBackend(root=Path("/")))
    payload = b"row\n" * 100
    zip_name = "s" * 238 + ".jsonl.zst"
    raw_name = zip_name + ".raw"  # 252 bytes
    _write_compressed(tmp_path / zip_name, payload, "zstd")
    zip_meta = ShardFile(
        basename=zip_name, bytes=(tmp_path / zip_name).stat().st_size, hashes={}
    )
    loc = _mk_locator(
        str(tmp_path),
        basename=raw_name,
        bytes=len(payload),
        zip_meta=zip_meta,
        compression="zstd",
    )
    assert resolver.resolve(loc).raw.path.read_bytes() == payload


def test_direct_resolver_surfaces_decoder_errors(tmp_path: Path) -> None:
    """A corrupt archive raises the decoder's own error, not a retry wrapper."""
    resolver = DirectResolver(LocalFSBackend(root=Path("/")))
    payload = b"row\n" * 100_000
    zip_path = tmp_path / "shard.zst"
    zip_path.write_bytes(zstd.compress(payload)[:-5])
    zip_meta = ShardFile(basename="shard.zst", bytes=zip_path.stat().st_size, hashes={})
    loc = _mk_locator(
        str(tmp_path),
        basename="shard.raw",
        bytes=len(payload),
        zip_meta=zip_meta,
        compression="zstd",
    )
    with pytest.raises(EOFError):
        resolver.resolve(loc)
    assert [p.name for p in tmp_path.iterdir()] == ["shard.zst"]


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
    zipf = tmp_path / "archive.br"
    zipf.write_bytes(b"br")
    zip_meta = ShardFile(basename="archive.br", bytes=zipf.stat().st_size, hashes={})
    loc = _mk_locator(
        str(tmp_path),
        basename="raw.bin",
        bytes=zipf.stat().st_size,
        zip_meta=zip_meta,
        compression="br",
    )
    with pytest.raises(ValueError, match="Unsupported compression"):
        _ = resolver.resolve(loc)


def _stub_decompress(
    monkeypatch: pytest.MonkeyPatch, outputs: list[bytes], attempts: dict[str, int]
) -> None:
    """Replace decompression with one that writes ``outputs[attempt]``."""

    def _fake(src: Path, dst: Path, compression: str) -> None:
        dst.write_bytes(outputs[min(attempts["count"], len(outputs) - 1)])
        attempts["count"] += 1

    monkeypatch.setattr(direct_mod, "decompress_file", _fake)


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
    _stub_decompress(monkeypatch, [payload], attempts)

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

    attempts = {"count": 0}
    # Broken decompression that yields nothing, twice, then the real payload.
    _stub_decompress(monkeypatch, [b"", b"", payload], attempts)

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

    attempts = {"count": 0}
    _stub_decompress(monkeypatch, [b""], attempts)  # always empty

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


class _NoIOBackend(LocalFSBackend):
    def __getattribute__(self, name: str) -> object:
        if name in {"open", "read_range", "stat", "exists", "download"}:
            raise AssertionError(f"Remote resolve did IO through {name}")
        return super().__getattribute__(name)


def test_direct_resolver_gives_remote_ref_without_io() -> None:
    remote = _NoIOBackend(root=Path("/"))
    resolver = DirectResolver(LocalFSBackend(root=Path("/")), remote_storage=remote)
    loc = _mk_locator("s3://bucket/data/", basename="part.bin", bytes=12)

    ref = resolver.resolve(loc)
    assert isinstance(ref, RemoteShardRef)
    assert ref.storage is remote
    assert ref.path == "s3://bucket/data/part.bin"
    assert ref.bytes == 12
    assert ref.cache_hit is False


def test_direct_resolver_refuses_remote_shards_it_cannot_read_in_place() -> None:
    local = LocalFSBackend(root=Path("/"))
    loc = _mk_locator("s3://bucket/data", basename="part.bin", bytes=12)
    with pytest.raises(ValueError, match="cache.enabled=True"):
        DirectResolver(local).resolve(loc)

    resolver = DirectResolver(local, remote_storage=local, validate_hash="xxh64")
    compressed = _mk_locator(
        "s3://bucket/data",
        basename="part.bin",
        bytes=12,
        zip_meta=ShardFile(basename="part.bin.zstd", bytes=5, hashes={}),
        compression="zstd",
    )
    with pytest.raises(ValueError, match="Compressed"):
        resolver.resolve(compressed)

    hashed = _mk_locator(
        "s3://bucket/data", basename="part.bin", bytes=12, hashes={"xxh64": "1"}
    )
    with pytest.raises(ValueError, match="validate_hash"):
        resolver.resolve(hashed)

    # Without a known hash there is nothing to validate.
    assert isinstance(resolver.resolve(loc), RemoteShardRef)
