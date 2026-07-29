# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Sync cloud-mirrored tokenizers to a local cache.

``AutoTokenizer.from_pretrained`` only understands HF Hub ids and local
filesystem paths. :func:`resolve_tokenizer_id` syncs an ``s3://`` /
``gs://`` / ``gcs://`` prefix to a deterministic local directory and
returns that path. The internal ``TokenizeText`` op calls it automatically
for cloud URIs.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from zephon._internal.io.storage import RouterStorageBackend

logger = logging.getLogger(__name__)


CLOUD_TOKENIZER_SCHEMES: frozenset[str] = frozenset({"s3", "gs", "gcs"})
TOKENIZER_CACHE_DIR_ENV: str = "ZEPHON_TOKENIZER_CACHE_DIR"
_DEFAULT_CACHE_ROOT: str = "~/.cache/zephon/tokenizers"


class CloudTokenizerError(RuntimeError):
    """Base error for cloud tokenizer synchronization failures."""


class PermanentCloudTokenizerError(CloudTokenizerError):
    """Cloud tokenizer error that should not be retried."""


class TransientCloudTokenizerError(CloudTokenizerError):
    """Cloud tokenizer error that may succeed on retry."""


def is_cloud_tokenizer_uri(tokenizer_id: str | None) -> bool:
    """Return True for ``s3://`` / ``gs://`` / ``gcs://`` URIs."""
    if not tokenizer_id:
        return False
    return urlparse(tokenizer_id).scheme in CLOUD_TOKENIZER_SCHEMES


def tokenizer_cache_root() -> Path:
    """Return ``$ZEPHON_TOKENIZER_CACHE_DIR`` or ``~/.cache/zephon/tokenizers``."""
    return Path(
        os.environ.get(TOKENIZER_CACHE_DIR_ENV, _DEFAULT_CACHE_ROOT)
    ).expanduser()


def _local_dir_for(scheme: str, bucket: str, prefix: str) -> Path:
    return tokenizer_cache_root() / scheme / bucket / prefix.rstrip("/")


def _validate_cache_prefix(prefix: str, tokenizer_id: str) -> None:
    """Reject cloud prefixes that would be unsafe as local cache paths."""
    trimmed = prefix.rstrip("/")
    if "\x00" in prefix:
        raise ValueError(f"NUL byte in cloud tokenizer prefix under {tokenizer_id!r}.")
    if any(part in {".", ".."} for part in trimmed.split("/")):
        raise ValueError(
            f"Unsafe cloud tokenizer prefix {prefix!r} under {tokenizer_id!r} "
            "(refusing to write outside cache)."
        )


def _credential_hint(scheme: str) -> str:
    if scheme == "s3":
        return (
            "Common causes: (1) AWS credentials missing/expired "
            "(set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or run "
            "`aws sso login`), (2) wrong S3 URI, "
            "(3) IAM access denied."
        )
    return (
        "Common causes: (1) GCS credentials missing "
        "(set GOOGLE_APPLICATION_CREDENTIALS or run "
        "`gcloud auth application-default login`), "
        "(2) wrong GCS URI, (3) IAM access denied."
    )


def _validate_relative_path(rel: str, tokenizer_id: str) -> None:
    """Reject listing entries that would escape the cache directory.

    Object-store keys are opaque strings — a polluted bucket may publish
    ``../foo`` or ``/etc/passwd`` and a naive join would write anywhere.
    """
    if not rel:
        raise ValueError(
            f"Empty relative path under {tokenizer_id!r} (refusing to write)."
        )
    if "\x00" in rel:
        raise ValueError(f"NUL byte in relative path {rel!r} under {tokenizer_id!r}.")
    if rel.startswith("/"):
        raise ValueError(
            f"Absolute relative path {rel!r} under {tokenizer_id!r} "
            f"(refusing to write outside cache)."
        )
    if any(part == ".." for part in rel.split("/")):
        raise ValueError(
            f"Parent-traversal segment in relative path {rel!r} under "
            f"{tokenizer_id!r} (refusing to write outside cache)."
        )


def resolve_tokenizer_id(tokenizer_id: str | None) -> str | None:
    """Resolve a tokenizer id, syncing cloud mirrors to a local cache.

    HF Hub ids, local paths, ``"__fallback__"`` and ``None`` are returned
    unchanged. Cloud URIs (``s3://`` / ``gs://`` / ``gcs://``) are synced
    to ``<cache_root>/<scheme>/<bucket>/<prefix>`` and that local path is
    returned. Files whose local size already matches the remote size are
    skipped.

    Args:
        tokenizer_id: HF Hub id, local path, cloud URI,
            ``"__fallback__"`` or ``None``.

    Returns:
        The resolved local path for cloud URIs; ``tokenizer_id`` unchanged
        otherwise.

    Raises:
        ValueError: if the cloud URI is malformed or the listing contains
            a path-traversal entry.
        RuntimeError: if listing fails, the prefix is empty, or any
            download fails.
    """
    if not is_cloud_tokenizer_uri(tokenizer_id):
        return tokenizer_id
    assert tokenizer_id is not None

    parsed = urlparse(tokenizer_id)
    scheme = parsed.scheme
    bucket = parsed.netloc
    # Leave the prefix verbatim — object-store keys are arbitrary byte
    # strings and percent-encoding has no special meaning at this layer.
    prefix = parsed.path.lstrip("/")

    if not bucket or not prefix:
        raise ValueError(
            f"Malformed cloud tokenizer URI {tokenizer_id!r}; "
            f"expected {scheme}://bucket/prefix/"
        )
    _validate_cache_prefix(prefix, tokenizer_id)

    storage = RouterStorageBackend()
    listing_uri = f"{scheme}://{bucket}/{prefix.rstrip('/')}/"
    try:
        files = list(storage.walk(listing_uri))
    except Exception as exc:
        raise TransientCloudTokenizerError(
            f"Failed to list cloud tokenizer mirror {tokenizer_id!r}. "
            f"Underlying error: {type(exc).__name__}: {exc}. "
            f"{_credential_hint(scheme)}"
        ) from exc

    if not files:
        raise PermanentCloudTokenizerError(
            f"Cloud tokenizer prefix {tokenizer_id!r} is empty; check "
            f"the URI and IAM permissions."
        )

    local_dir = _local_dir_for(scheme, bucket, prefix)
    cache_root = tokenizer_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_root_resolved = cache_root.resolve()
    local_dir_resolved = local_dir.resolve()
    if not local_dir_resolved.is_relative_to(cache_root_resolved):
        raise ValueError(
            f"Refusing to use cache directory {local_dir} outside cache root "
            f"{cache_root_resolved} for {tokenizer_id!r}."
        )
    local_dir.mkdir(parents=True, exist_ok=True)
    local_dir_resolved = local_dir.resolve()
    if not local_dir_resolved.is_relative_to(cache_root_resolved):
        raise ValueError(
            f"Refusing to use cache directory {local_dir} outside cache root "
            f"{cache_root_resolved} for {tokenizer_id!r}."
        )
    logger.info(
        "Resolving cloud tokenizer mirror %s -> %s (%d remote file(s); checking cache)",
        tokenizer_id,
        local_dir,
        len(files),
    )
    downloaded = 0
    for rel, remote_size in files:
        _validate_relative_path(rel, tokenizer_id)
        local_path = local_dir / rel
        # Defense-in-depth against symlinks pre-existing in the cache.
        if not local_path.resolve().is_relative_to(local_dir_resolved):
            raise ValueError(
                f"Refusing to write {local_path} outside cache root "
                f"{local_dir_resolved} (computed from {rel!r} under "
                f"{tokenizer_id!r})."
            )
        if local_path.exists() and local_path.stat().st_size == remote_size:
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic per-file rename. Concurrent multi-process writers can't
        # corrupt a single file, but a reader can still observe a
        # half-populated directory mid-sync — callers needing dir-level
        # atomicity must coordinate externally.
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{local_path.name}.", suffix=".part", dir=local_path.parent
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            storage.download(listing_uri + rel, str(tmp_path))
            os.replace(tmp_path, local_path)
            downloaded += 1
        except Exception as exc:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            raise TransientCloudTokenizerError(
                f"Failed to sync cloud tokenizer mirror "
                f"{tokenizer_id!r} to {local_dir}. Underlying error: "
                f"{type(exc).__name__}: {exc}. {_credential_hint(scheme)}"
            ) from exc

    logger.info(
        "Resolved cloud tokenizer mirror %s -> %s (%d new file(s), %d cached)",
        tokenizer_id,
        local_dir,
        downloaded,
        len(files) - downloaded,
    )
    return str(local_dir)


@retry(
    wait=wait_random_exponential(multiplier=2, max=15),
    stop=stop_after_attempt(5),
    retry=retry_if_not_exception_type((ValueError, PermanentCloudTokenizerError)),
    reraise=True,
)
def resolve_tokenizer_id_with_retry(tokenizer_id: str | None) -> str | None:
    """Retrying wrapper around :func:`resolve_tokenizer_id`.

    Retries transient sync failures with exponential backoff; ``ValueError``
    (malformed URI) and :class:`PermanentCloudTokenizerError` are raised
    immediately.
    """
    return resolve_tokenizer_id(tokenizer_id)


__all__ = [
    "CloudTokenizerError",
    "CLOUD_TOKENIZER_SCHEMES",
    "PermanentCloudTokenizerError",
    "TOKENIZER_CACHE_DIR_ENV",
    "TransientCloudTokenizerError",
    "is_cloud_tokenizer_uri",
    "resolve_tokenizer_id",
    "resolve_tokenizer_id_with_retry",
    "tokenizer_cache_root",
]
