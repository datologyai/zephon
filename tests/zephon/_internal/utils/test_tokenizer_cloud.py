# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Tests for cloud tokenizer URI resolution."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from tenacity.wait import wait_none

from zephon._internal.utils.tokenizer_cloud import (
    CLOUD_TOKENIZER_SCHEMES,
    TOKENIZER_CACHE_DIR_ENV,
    PermanentCloudTokenizerError,
    TransientCloudTokenizerError,
    is_cloud_tokenizer_uri,
    resolve_tokenizer_id,
    resolve_tokenizer_id_with_retry,
    tokenizer_cache_root,
)

# ---------- helpers ---------- #


class _FakeObstoreChunk:
    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def __iter__(self):
        return iter(self._items)


def _fake_obstore_list(prefix_to_objects: dict[str, list[dict]]):
    def _list(store, prefix: str):
        del store
        return [_FakeObstoreChunk(prefix_to_objects.get(prefix, []))]

    return _list


def _objects(prefix: str, names_and_sizes: list[tuple[str, int]]) -> list[dict]:
    return [{"path": prefix + name, "size": size} for name, size in names_and_sizes]


def _patch_walk(uri_to_files: dict[str, list[tuple[str, int]]]):
    canonicalised = {f"{u.rstrip('/')}/": list(f) for u, f in uri_to_files.items()}

    def fresh_walk(self, path):
        del self
        return iter(canonicalised.get(path, []))

    return patch(
        "zephon._internal.io.storage.router.RouterStorageBackend.walk", new=fresh_walk
    )


def _patch_download(handler):
    def _wrapped(self, src, dst, timeout=None):
        del self, timeout
        return handler(src, dst)

    return patch(
        "zephon._internal.io.storage.router.RouterStorageBackend.download", new=_wrapped
    )


# ---------- pass-through cases ---------- #


@pytest.mark.parametrize(
    "tokenizer_id",
    [
        None,
        "",
        "meta-llama/Llama-3.2-1B",
        "unsloth/Llama-3.2-1B",
        "__fallback__",
        "/opt/tokenizers/Llama-3.2-1B",
        "./relative/path",
    ],
)
def test_resolve_passthrough(tokenizer_id: str | None) -> None:
    """Non-cloud ids must be returned verbatim with zero I/O."""
    result = resolve_tokenizer_id(tokenizer_id)
    assert result == tokenizer_id


@pytest.mark.parametrize(
    "tokenizer_id,expected",
    [
        (None, False),
        ("", False),
        ("meta-llama/Llama-3.2-1B", False),
        ("/local/path", False),
        ("__fallback__", False),
        ("s3://bucket/prefix", True),
        ("s3://bucket/prefix/", True),
        ("gs://bucket/prefix/", True),
        ("gcs://bucket/prefix/", True),
        ("https://example.com/path", False),
        ("ftp://example.com/path", False),
    ],
)
def test_is_cloud_tokenizer_uri(tokenizer_id: str | None, expected: bool) -> None:
    assert is_cloud_tokenizer_uri(tokenizer_id) is expected


def test_cloud_schemes_are_lower_case() -> None:
    """Guard against accidental upper-case entries (urlparse lower-cases scheme)."""
    for s in CLOUD_TOKENIZER_SCHEMES:
        assert s == s.lower()


# ---------- cache root resolution ---------- #


def test_cache_root_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKENIZER_CACHE_DIR_ENV, raising=False)
    assert tokenizer_cache_root() == Path(
        os.path.expanduser("~/.cache/zephon/tokenizers")
    )


def test_cache_root_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))
    assert tokenizer_cache_root() == tmp_path


def test_cache_root_env_expands_tilde(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, "~/custom-cache")
    root = tokenizer_cache_root()
    assert "~" not in str(root)
    assert root == Path(os.path.expanduser("~/custom-cache"))


# ---------- malformed URIs ---------- #


@pytest.mark.parametrize(
    "uri",
    [
        "s3://",  # no bucket, no prefix
        "s3:///prefix/",  # no bucket
        "s3://bucket",  # no prefix
        "s3://bucket/",  # bucket only, no key prefix
        "gs://bucket",  # no prefix (gcs branch)
    ],
)
def test_malformed_uri_raises_value_error(
    uri: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))
    with pytest.raises(ValueError, match="Malformed"):
        resolve_tokenizer_id(uri)


# ---------- path-traversal defense ---------- #


@pytest.mark.parametrize(
    "rel",
    [
        "../etc/passwd",
        "subdir/../../etc/passwd",
        "/etc/passwd",
        "..",
        "subdir/..",
    ],
)
def test_resolve_rejects_path_traversal(
    rel: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A malicious or polluted bucket must not be able to write outside cache."""
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))
    uri = "s3://fake-bucket/tok/"
    files = [(rel, 4), ("tokenizer.json", 4)]

    def fake_download(src, dst):  # pragma: no cover - we never get here
        Path(dst).write_bytes(b"data")

    with _patch_walk({uri: files}), _patch_download(fake_download):
        with pytest.raises(ValueError, match="cache"):
            resolve_tokenizer_id(uri)


@pytest.mark.parametrize(
    "uri",
    [
        "s3://fake-bucket/../../../escape/",
        "s3://fake-bucket/tok/../../escape/",
        "s3://fake-bucket/./tok/",
    ],
)
def test_resolve_rejects_unsafe_uri_prefix_before_listing(
    uri: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cloud prefix itself must not escape the local cache root."""
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))

    def explode_walk(self, path):  # pragma: no cover - validation should run first
        raise AssertionError(f"walk should not be called for {path}")

    with patch(
        "zephon._internal.io.storage.router.RouterStorageBackend.walk", new=explode_walk
    ):
        with pytest.raises(ValueError, match="prefix"):
            resolve_tokenizer_id(uri)

    assert list(tmp_path.iterdir()) == []


def test_resolve_rejects_symlink_traversal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Symlinks pre-existing in the cache must not let a write escape it."""
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))
    uri = "s3://fake-bucket/tok/"
    cache_dir = tmp_path / "s3" / "fake-bucket" / "tok"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path.parent / "escape-target"
    outside.mkdir(exist_ok=True)
    (cache_dir / "evil").symlink_to(outside, target_is_directory=True)

    files = [("evil/loot.txt", 4)]

    def fake_download(src, dst):  # pragma: no cover
        Path(dst).write_bytes(b"data")

    with _patch_walk({uri: files}), _patch_download(fake_download):
        with pytest.raises(ValueError, match="outside cache root"):
            resolve_tokenizer_id(uri)


# ---------- happy path: full sync ---------- #


def test_resolve_syncs_flat_tokenizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))
    bucket = "fake-bucket"
    prefix = "tokenizers/Llama-3.2-1B/"
    uri = f"s3://{bucket}/{prefix}"
    files = [
        ("tokenizer.json", 32),
        ("tokenizer_config.json", 16),
        ("special_tokens_map.json", 8),
        ("config.json", 4),
    ]

    downloads: list[tuple[str, str]] = []

    def fake_download(src, dst):
        downloads.append((src, dst))
        size = next(s for name, s in files if src.endswith(name))
        Path(dst).write_bytes(b"\x00" * size)

    with _patch_walk({uri: files}), _patch_download(fake_download):
        local = resolve_tokenizer_id(uri)

    assert local is not None
    assert local == str(tmp_path / "s3" / bucket / prefix.rstrip("/"))
    assert len(downloads) == len(files)
    for name, size in files:
        local_file = Path(local) / name
        assert local_file.exists(), f"{name} missing"
        assert local_file.stat().st_size == size


def test_resolve_skips_already_cached_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Files whose local size matches the remote listing must not redownload."""
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))
    bucket = "fake-bucket"
    prefix = "tok/"
    uri = f"s3://{bucket}/{prefix}"
    files = [("tokenizer.json", 1024), ("config.json", 256)]

    local_dir = tmp_path / "s3" / bucket / prefix.rstrip("/")
    local_dir.mkdir(parents=True, exist_ok=True)
    for name, size in files:
        (local_dir / name).write_bytes(b"\x00" * size)

    download_calls: list[tuple[str, str]] = []

    def fake_download(src, dst):
        download_calls.append((src, dst))

    with _patch_walk({uri: files}), _patch_download(fake_download):
        local = resolve_tokenizer_id(uri)

    assert local == str(local_dir)
    assert download_calls == [], "Already-cached files must not be re-downloaded"


def test_resolve_redownloads_when_size_mismatches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If local size differs from remote size, the file must be redownloaded."""
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))
    bucket = "fake-bucket"
    prefix = "tok/"
    uri = f"s3://{bucket}/{prefix}"
    files = [("tokenizer.json", 1024)]

    local_dir = tmp_path / "s3" / bucket / prefix.rstrip("/")
    local_dir.mkdir(parents=True, exist_ok=True)
    # Partial / corrupted local file: 100 bytes vs remote 1024.
    (local_dir / "tokenizer.json").write_bytes(b"\x00" * 100)

    download_calls: list[tuple[str, str]] = []

    def fake_download(src, dst):
        download_calls.append((src, dst))
        Path(dst).write_bytes(b"\x00" * 1024)

    with _patch_walk({uri: files}), _patch_download(fake_download):
        local = resolve_tokenizer_id(uri)

    assert local == str(local_dir)
    assert len(download_calls) == 1
    assert (local_dir / "tokenizer.json").stat().st_size == 1024


def test_resolve_preserves_nested_subdirectories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Some HF tokenizers ship with subdirectories (e.g. sentence-transformers)."""
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))
    bucket = "fake-bucket"
    prefix = "tok/"
    uri = f"s3://{bucket}/{prefix}"
    files = [
        ("tokenizer.json", 8),
        ("1_Pooling/config.json", 4),
        ("2_Dense/pytorch_model.bin", 16),
    ]

    def fake_download(src, dst):
        size = next(s for name, s in files if src.endswith(name))
        Path(dst).write_bytes(b"\x00" * size)

    with _patch_walk({uri: files}), _patch_download(fake_download):
        local = resolve_tokenizer_id(uri)

    assert local is not None
    for name, size in files:
        local_file = Path(local) / name
        assert local_file.exists(), f"{name} missing from {local}"
        assert local_file.stat().st_size == size
    # Subdirectory was created.
    assert (Path(local) / "1_Pooling").is_dir()
    assert (Path(local) / "2_Dense").is_dir()


# ---------- error wrapping ---------- #


def test_empty_prefix_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))
    with _patch_walk({"s3://fake-bucket/empty/prefix/": []}):
        with pytest.raises(PermanentCloudTokenizerError, match="empty"):
            resolve_tokenizer_id("s3://fake-bucket/empty/prefix/")


def test_empty_prefix_does_not_create_local_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Misconfigured/typo'd prefixes shouldn't litter empty dirs in the cache."""
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))
    with _patch_walk({"s3://fake-bucket/empty/prefix/": []}):
        with pytest.raises(RuntimeError):
            resolve_tokenizer_id("s3://fake-bucket/empty/prefix/")

    # Cache dir for this prefix should not exist (validation happens before mkdir).
    assert not (tmp_path / "s3" / "fake-bucket" / "empty" / "prefix").exists()


def test_listing_error_wrapped_with_aws_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))

    def explode_walk(self, path):
        raise RuntimeError("AccessDenied")

    with patch(
        "zephon._internal.io.storage.router.RouterStorageBackend.walk", new=explode_walk
    ):
        with pytest.raises(TransientCloudTokenizerError) as ctx:
            resolve_tokenizer_id("s3://locked-bucket/some/prefix/")

    msg = str(ctx.value)
    assert "Failed to list" in msg
    assert "AWS credentials" in msg
    assert "AccessDenied" in msg


def test_listing_error_wrapped_with_gcs_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))

    def explode_walk(self, path):
        raise RuntimeError("permission denied")

    with patch(
        "zephon._internal.io.storage.router.RouterStorageBackend.walk", new=explode_walk
    ):
        with pytest.raises(TransientCloudTokenizerError) as ctx:
            resolve_tokenizer_id("gs://locked-bucket/some/prefix/")

    msg = str(ctx.value)
    assert "Failed to list" in msg
    assert "GCS credentials" in msg
    assert "AWS" not in msg


def test_download_error_wrapped_and_temp_cleaned_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Download failure surfaces a RuntimeError and leaves no .part files behind."""
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))
    bucket = "fake-bucket"
    prefix = "tok/"
    uri = f"s3://{bucket}/{prefix}"
    files = [("subdir/tokenizer.json", 100)]

    def fake_download(src, dst):
        raise OSError("network borked")

    with _patch_walk({uri: files}), _patch_download(fake_download):
        with pytest.raises(TransientCloudTokenizerError, match="Failed to sync"):
            resolve_tokenizer_id(uri)

    local_dir = tmp_path / "s3" / bucket / prefix.rstrip("/")
    part_files = list(local_dir.rglob("*.part"))
    assert part_files == [], f".part files leaked on failure: {part_files}"


# ---------- concurrency-safety ---------- #


def test_concurrent_resolution_same_uri_is_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Concurrent resolves of the same URI must all succeed without corruption.

    Patches go on the main thread because ``unittest.mock.patch`` is not
    thread-safe — entering it from N threads can leak state across tests.
    """
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))
    bucket = "fake-bucket"
    prefix = "tok/"
    uri = f"s3://{bucket}/{prefix}"
    files = [("tokenizer.json", 64), ("config.json", 16)]

    barrier = threading.Barrier(4)

    def fake_download(src, dst):
        try:
            barrier.wait(timeout=2.0)
        except threading.BrokenBarrierError:
            pass
        size = next(s for name, s in files if src.endswith(name))
        Path(dst).write_bytes(b"\x00" * size)

    results: list[str | None] = []
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            results.append(resolve_tokenizer_id(uri))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    with _patch_walk({uri: files}), _patch_download(fake_download):
        threads = [threading.Thread(target=_run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert not errors, f"Concurrent resolves errored: {errors}"
    assert all(r is not None for r in results)
    assert len(set(results)) == 1, "All threads must agree on the path"

    local_dir = Path(results[0])  # type: ignore[arg-type]
    for name, size in files:
        local_file = local_dir / name
        assert local_file.exists()
        assert local_file.stat().st_size == size
    # No .part residue.
    assert list(local_dir.glob("*.part*")) == []


# ---------- idempotency / second-call optimization ---------- #


def test_second_call_logs_zero_new_files(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """A repeat call to ``resolve_tokenizer_id`` is cheap (no downloads)."""
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))
    caplog.set_level(logging.INFO, logger="zephon._internal.utils.tokenizer_cloud")

    bucket = "fake-bucket"
    prefix = "tok/"
    uri = f"s3://{bucket}/{prefix}"
    files = [("tokenizer.json", 12)]

    def fake_download(src, dst):
        Path(dst).write_bytes(b"\x00" * 12)

    with _patch_walk({uri: files}), _patch_download(fake_download):
        first = resolve_tokenizer_id(uri)

    caplog.clear()

    download_count = 0

    def counting_download(src, dst):
        nonlocal download_count
        download_count += 1

    with _patch_walk({uri: files}), _patch_download(counting_download):
        second = resolve_tokenizer_id(uri)

    assert first == second
    assert download_count == 0
    assert any("1 cached" in r.message for r in caplog.records)


# ---------- real walk-boundary test ---------- #


def test_real_walk_through_obstore_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end through ``RouterStorageBackend.walk`` → obstore listing.

    Patches at ``obstore.list`` rather than the resolver's helpers, so the
    listing→walk→resolve path actually executes — that's the surface most
    likely to break when the storage layer changes.
    """
    monkeypatch.setenv(TOKENIZER_CACHE_DIR_ENV, str(tmp_path))
    bucket = "fake-bucket"
    prefix = "tok/"
    uri = f"s3://{bucket}/{prefix}"

    # Mix of: in-prefix files, a folder marker, and an out-of-prefix object.
    raw_objects = [
        {"path": "tok/tokenizer.json", "size": 8},
        {"path": "tok/subdir/config.json", "size": 4},
        {"path": "tok/subdir/", "size": 0},
        {"path": "other/leak.json", "size": 99},
    ]

    downloads: list[str] = []

    def fake_obstore_list(store, prefix):
        del store
        if prefix == "tok/":
            return [_FakeObstoreChunk(raw_objects)]
        return [_FakeObstoreChunk([])]

    def fake_download(self, src, dst, timeout=None):
        del self, timeout
        downloads.append(src)
        # Size derived from the listing entry we expect.
        if src.endswith("tokenizer.json"):
            Path(dst).write_bytes(b"\x00" * 8)
        elif src.endswith("config.json"):
            Path(dst).write_bytes(b"\x00" * 4)
        else:  # pragma: no cover - the leak.json case must never reach here
            raise AssertionError(f"Unexpected download URL: {src}")

    class _StubStore:  # minimal interface for the fake list
        pass

    with (
        patch("obstore.list", side_effect=fake_obstore_list),
        patch(
            "zephon._internal.io.storage.s3.S3Backend._get_store",
            return_value=_StubStore(),
        ),
        patch(
            "zephon._internal.io.storage.router.RouterStorageBackend.download",
            new=fake_download,
        ),
    ):
        local = resolve_tokenizer_id(uri)

    assert local is not None
    assert (Path(local) / "tokenizer.json").stat().st_size == 8
    assert (Path(local) / "subdir" / "config.json").stat().st_size == 4
    assert not any(src.endswith("subdir/") for src in downloads)
    assert not any("other/leak.json" in src for src in downloads)
    assert len(downloads) == 2


# ---------- resolve_tokenizer_id_with_retry ---------- #
# The wrapper resolves the module-global ``resolve_tokenizer_id`` at call time,
# so tests must patch it here in ``tokenizer_cloud`` (not wherever the wrapper
# is imported). The retry envelope is built at import, so backoff is dropped by
# mutating the controller rather than patching ``wait_random_exponential``.


def test_resolve_with_retry_does_not_retry_permanent_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deterministic resolver failures must surface without backoff retries."""
    attempts = 0

    def permanent_resolve(tokenizer_id: str | None) -> str | None:
        nonlocal attempts
        attempts += 1
        raise PermanentCloudTokenizerError(f"empty prefix: {tokenizer_id}")

    monkeypatch.setattr(
        "zephon._internal.utils.tokenizer_cloud.resolve_tokenizer_id", permanent_resolve
    )

    with pytest.raises(PermanentCloudTokenizerError, match="empty prefix"):
        resolve_tokenizer_id_with_retry("s3://fake-bucket/empty/")

    assert attempts == 1


def test_resolve_with_retry_retries_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient sync failures are retried; the second attempt succeeds."""
    attempts = 0

    def flaky_resolve(tokenizer_id: str | None) -> str | None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TransientCloudTokenizerError("simulated transient S3 5xx")
        return "/local/resolved"

    monkeypatch.setattr(
        "zephon._internal.utils.tokenizer_cloud.resolve_tokenizer_id", flaky_resolve
    )
    monkeypatch.setattr(resolve_tokenizer_id_with_retry.retry, "wait", wait_none())

    assert (
        resolve_tokenizer_id_with_retry("s3://fake-bucket/prefix/") == "/local/resolved"
    )
    assert attempts == 2
