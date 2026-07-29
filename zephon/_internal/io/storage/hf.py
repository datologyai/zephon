"""HuggingFace dataset storage backend.

Treats an ``hf://org/name[@rev]/[config/]split`` URI as a virtual directory of
parquet shards backed by the Datasets Server ``/parquet`` endpoint:

- ``listdir`` issues one HTTP call to ``/parquet`` and returns the parquet
  shard filenames for the requested split.
- ``stat`` / ``read_range`` operate on the per-file HTTPS URLs returned by
  ``/parquet`` (``huggingface.co`` hosts speak HTTP range GETs), reusing the
  cached listing so there are no per-file HEAD requests.
- ``download`` delegates to ``huggingface_hub.file_download.http_get`` for
  HF's redirect/retry/resume handling

The backend is registered for the ``hf://`` scheme in
:class:`zephon._internal.io.storage.router.RouterStorageBackend`, which lets
:class:`zephon._internal.io.formats.parquet.ParquetFormat` consume HF datasets without
any HF-specific code paths in ``Dataset.from_path`` or the format handlers.

Auth: bearer token from ``huggingface_hub.HfFolder.get_token()`` if available,
otherwise the ``HF_TOKEN`` environment variable.

Revision semantics:

- Default (``main``, i.e. no ``@<rev>``): we use the URLs ``/parquet``
  returns, which point at the auto-converted ``refs/convert/parquet``
  branch. This is the path that always has parquet, regardless of the
  dataset's native format.
- Explicit ``@<rev>`` (a tag, branch, or commit SHA): we extract the
  per-shard path from ``/parquet``'s URL and rebuild a resolve URL via
  ``huggingface_hub.hf_hub_url(repo_id, path, revision=<rev>)``. For
  parquet-native repos that maintain a stable directory layout this
  delivers reproducible per-revision pinning. For non-parquet-native
  repos the rebuilt URL will 404 at read/download time (those revisions
  don't have parquet at the same paths), surfacing as
  ``FileNotFoundError`` — honest failure instead of silently serving the
  auto-converted bytes under a mismatched cache key.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Mapping

from ._hf_uri import HF_URI_SCHEME, HFUriParts, parse_hf_uri
from ._utils import OpenViaDownloadMixin

logger = logging.getLogger(__name__)

_DATASETS_SERVER_PARQUET_URL = "https://datasets-server.huggingface.co/parquet"
_HTTP_TIMEOUT_SECONDS = 120.0

# Retry transient errors on the HF Datasets Server and CDN. The /parquet
# endpoint returns 503 while a dataset is being converted, and the CDN
# occasionally returns 429 under high concurrency.
_RETRY_TOTAL = 4
_RETRY_BACKOFF_FACTOR = 0.5
_RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)

# ``partial=true`` means the Datasets Server only converted part of the
# dataset, so the shard listing is incomplete. Silently training on a
# truncated dataset is the foot-gun the rewrite is meant to avoid; refuse
# by default and let the user opt in if they explicitly want a partial view.
_ALLOW_PARTIAL_ENV = "ZEPHON_HF_ALLOW_PARTIAL"


@dataclass(frozen=True)
class _HFShard:
    """A parquet shard advertised by the Datasets Server."""

    filename: str
    url: str
    size: int


_TOKEN_UNSET: Any = object()


class HFBackend(OpenViaDownloadMixin):
    """Storage backend that reads parquet shards from HuggingFace datasets.

    All IO is just-in-time: there is no pre-download to disk and no symlink
    view. The first call into the backend for a given
    ``(repo_id, revision, config, split)`` populates a per-instance listing
    cache; subsequent ``stat`` / ``read_range`` / ``download`` calls reuse the
    cached URL and size.
    """

    def __init__(self) -> None:
        self._listing_cache: dict[
            tuple[str, str, str | None, str], dict[str, _HFShard]
        ] = {}
        self._cache_lock = threading.Lock()
        self._token: Any = _TOKEN_UNSET
        self._token_lock = threading.Lock()
        self._session: Any = None
        self._session_lock = threading.Lock()

    def _get_session(self) -> Any:
        """Return a lazily-built ``requests.Session`` with retry on 429/5xx."""
        if self._session is not None:
            return self._session
        with self._session_lock:
            if self._session is None:
                try:
                    import requests
                    from requests.adapters import HTTPAdapter
                    from urllib3.util.retry import Retry
                except ImportError as exc:
                    raise ImportError(
                        "The `requests` library is required for hf:// URIs. "
                        "Install with: pip install zephon[hf]"
                    ) from exc

                retry = Retry(
                    total=_RETRY_TOTAL,
                    backoff_factor=_RETRY_BACKOFF_FACTOR,
                    status_forcelist=_RETRY_STATUS_FORCELIST,
                    allowed_methods=frozenset({"GET", "HEAD"}),
                    respect_retry_after_header=True,
                    raise_on_status=False,
                )
                adapter = HTTPAdapter(max_retries=retry)
                session = requests.Session()
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                self._session = session
        return self._session

    @staticmethod
    def _not_found_message(path: str) -> str:
        """404 error message, hinting at @<rev> path-layout mismatch."""
        try:
            parts, _ = HFBackend._split_path(path)
        except ValueError:
            parts = None
        if parts is not None and parts.revision != "main":
            return (
                f"hf:// object not found at @{parts.revision!r}: {path}. "
                f"The path used by /parquet (from the auto-converted "
                f"refs/convert/parquet branch) likely doesn't exist at "
                f"@{parts.revision}; that revision may use a different "
                "layout. Drop the @<revision> suffix to use the "
                "auto-converted view, or query the HF tree API to find the "
                "actual parquet paths at your revision."
            )
        return f"hf:// object not found: {path}"

    @staticmethod
    def _raise_for_http_error(resp: Any, path: str) -> None:
        """Map HTTP error responses to protocol-conforming exceptions."""
        status = resp.status_code
        if status == 404:
            raise FileNotFoundError(HFBackend._not_found_message(path))
        if status in (401, 403):
            raise PermissionError(
                f"hf:// access denied: {path}; "
                "set HF_TOKEN or run `huggingface-cli login`."
            )
        resp.raise_for_status()

    @staticmethod
    def _resolve_shard_url(repo_id: str, parquet_url: str, revision: str) -> str:
        """Return the URL to actually fetch a shard from.

        For the default revision (``main``), the URL returned by ``/parquet``
        (always pointing at the auto-converted ``refs/convert/parquet``
        branch) is the canonical source.

        For any other revision the user explicitly typed (``@v1.0``,
        ``@<sha>``), we rebuild the URL via ``huggingface_hub.hf_hub_url``
        so the shard fetch is pinned to that revision. This delivers real
        reproducibility for parquet-native repos; for non-parquet-native
        repos the URL will 404 at read/download time (the user revision
        doesn't have parquet at those paths), which surfaces honestly as
        ``FileNotFoundError`` rather than silently returning the
        auto-converted bytes under a misleading cache key.

        Path extraction: ``/parquet`` returns canonical resolve URLs of the
        form
        ``https://huggingface.co/datasets/{repo}/resolve/{encoded-rev}/{path}``
        (HF URL-encodes ``refs/convert/parquet`` as ``refs%2Fconvert%2Fparquet``
        in the path segment). We slice off the prefix and the embedded
        ref, and reuse the trailing ``{path}`` for the rebuilt URL on the
        assumption that the user-specified revision uses the same
        layout — which is true for parquet-native repos that maintain a
        stable directory structure.
        """
        if revision == "main":
            return parquet_url

        import urllib.parse as _urlparse

        parsed = _urlparse.urlparse(parquet_url)
        if parsed.netloc != "huggingface.co":
            raise RuntimeError(
                f"Cannot pin to revision {revision!r}: /parquet returned URL "
                f"{parquet_url!r} on an unexpected host. This usually means "
                "HF changed the URL shape; please report it. Omit "
                "@<revision> for the default view in the meantime."
            )
        resolve_prefix = f"/datasets/{repo_id}/resolve/"
        if not parsed.path.startswith(resolve_prefix):
            raise RuntimeError(
                f"Cannot pin to revision {revision!r}: /parquet URL "
                f"{parquet_url!r} doesn't match the expected resolve shape "
                f"{resolve_prefix}.... This usually means HF changed the "
                "URL format; please report it. Omit @<revision> for the "
                "default view in the meantime."
            )
        rest = parsed.path[len(resolve_prefix) :]
        if "/" not in rest:
            raise RuntimeError(
                f"Cannot pin to revision {revision!r}: /parquet URL "
                f"{parquet_url!r} has no path after the embedded ref."
            )
        # Discard the embedded ref; the user's revision replaces it.
        _, path_in_repo = rest.split("/", 1)

        try:
            from huggingface_hub import hf_hub_url

            return hf_hub_url(
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type="dataset",
                revision=revision,
            )
        except ImportError:
            # huggingface_hub absent (e.g. 3.14t CI). Build the resolve URL
            # manually — same shape as hf_hub_url produces.
            quoted_rev = _urlparse.quote(revision, safe="")
            return (
                f"https://huggingface.co/datasets/{repo_id}/resolve/"
                f"{quoted_rev}/{path_in_repo}"
            )

    @staticmethod
    def _split_matches(item: Mapping[str, Any], parts: HFUriParts) -> bool:
        if item.get("split") != parts.split:
            return False
        if parts.config is not None and item.get("config") != parts.config:
            return False
        return True

    @classmethod
    def _guard_against_incomplete_conversion(
        cls, payload: Mapping[str, Any], parts: HFUriParts
    ) -> None:
        """Refuse to silently serve a truncated split listing.

        ``/parquet`` exposes three independent failure signals:
        - ``partial=true``: dataset-wide conversion is still streaming and the
          listing may be missing later shards;
        - ``failed=[...]``: per-(config, split) entries the converter gave up
          on; the shard set for those splits is permanently incomplete;
        - ``pending=[...]``: per-(config, split) entries still being converted.

        We raise if any of those touch the requested split. ``failed`` is
        always fatal (waiting won't help). ``partial``/``pending`` can be
        opted into via ``ZEPHON_HF_ALLOW_PARTIAL=1`` for diagnostic runs.
        """
        failed = payload.get("failed") or []
        for item in failed:
            if cls._split_matches(item, parts):
                raise RuntimeError(
                    f"HuggingFace Datasets Server reports failed conversion "
                    f"for {parts.repo_id!r} split={parts.split!r}; the "
                    "parquet shard set is permanently incomplete for this "
                    "split. Pick a different split, or open an issue on the "
                    "dataset repo."
                )

        allow_partial = bool(os.environ.get(_ALLOW_PARTIAL_ENV))
        if allow_partial:
            return

        pending = payload.get("pending") or []
        for item in pending:
            if cls._split_matches(item, parts):
                raise RuntimeError(
                    f"HuggingFace Datasets Server reports pending conversion "
                    f"for {parts.repo_id!r} split={parts.split!r}; the shard "
                    "listing would be truncated. Retry later, or set "
                    f"{_ALLOW_PARTIAL_ENV}=1 to opt in to a partial view."
                )

        if payload.get("partial"):
            raise RuntimeError(
                f"HuggingFace Datasets Server reports partial=true for "
                f"{parts.repo_id!r} split={parts.split!r}; conversion is "
                "incomplete and the shard listing would be truncated. Retry "
                f"later, or set {_ALLOW_PARTIAL_ENV}=1 to opt in to a "
                "partial view."
            )

    def _get_token(self) -> str | None:
        if self._token is not _TOKEN_UNSET:
            return self._token
        with self._token_lock:
            if self._token is _TOKEN_UNSET:
                token: str | None = None
                try:
                    from huggingface_hub import HfFolder

                    token = HfFolder.get_token()
                except ImportError:
                    token = None
                self._token = token or os.environ.get("HF_TOKEN")
        return self._token

    def _auth_headers(self) -> dict[str, str]:
        token = self._get_token()
        return {"Authorization": f"Bearer {token}"} if token else {}

    @staticmethod
    def _split_path(path: str) -> tuple[HFUriParts, str | None]:
        """Return ``(dataset parts, optional filename)`` for an ``hf://`` path.

        File paths take the form ``hf://org/name[@rev]/[config/]split/<file>``;
        a trailing segment is treated as a filename when it contains a ``.``
        and the URI has at least four path segments after the scheme. This
        leaves the documented split/config grammar of :func:`parse_hf_uri`
        unchanged for dataset-level URIs.
        """
        if not path.startswith(HF_URI_SCHEME):
            raise ValueError(f"Not an hf:// URI: {path!r}")

        body = path[len(HF_URI_SCHEME) :].rstrip("/")
        segments = body.split("/") if body else []
        if segments and "." in segments[-1] and len(segments) >= 4:
            filename = segments[-1]
            dataset_uri = HF_URI_SCHEME + "/".join(segments[:-1])
            return parse_hf_uri(dataset_uri), filename
        return parse_hf_uri(path), None

    def _list_shards(self, parts: HFUriParts) -> dict[str, _HFShard]:
        key = (parts.repo_id, parts.revision, parts.config, parts.split)
        with self._cache_lock:
            cached = self._listing_cache.get(key)
            if cached is not None:
                return cached

        params: dict[str, str] = {"dataset": parts.repo_id}
        if parts.config is not None:
            params["config"] = parts.config

        session = self._get_session()
        resp = session.get(
            _DATASETS_SERVER_PARQUET_URL,
            params=params,
            headers=self._auth_headers(),
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        self._raise_for_http_error(
            resp,
            f"hf://{parts.repo_id}@{parts.revision}/{parts.config or ''}/{parts.split}",
        )
        payload = resp.json()

        self._guard_against_incomplete_conversion(payload, parts)

        shards: dict[str, _HFShard] = {}
        for item in payload.get("parquet_files", []):
            if item.get("split") != parts.split:
                continue
            if parts.config is not None and item.get("config") != parts.config:
                continue
            filename = item.get("filename")
            parquet_url = item.get("url")
            size = item.get("size")
            if not filename or not parquet_url:
                continue
            if filename in shards:
                # Same filename appearing under two different configs: refuse
                # rather than silently picking one. The user should pin a
                # config in the URI.
                raise ValueError(
                    f"Ambiguous filename {filename!r} in {parts.repo_id!r} "
                    f"split={parts.split!r}; specify a config in the hf:// URI."
                )
            url = self._resolve_shard_url(parts.repo_id, parquet_url, parts.revision)
            shards[filename] = _HFShard(
                filename=filename,
                url=url,
                size=int(size) if size is not None else 0,
            )

        if not shards:
            config_msg = f"config={parts.config!r}, " if parts.config else ""
            raise FileNotFoundError(
                f"No parquet shards for {config_msg}split={parts.split!r} "
                f"in {parts.repo_id!r}. The dataset may not expose parquet "
                "via the Datasets Server, or the split/config may be wrong."
            )

        with self._cache_lock:
            self._listing_cache.setdefault(key, shards)
            return self._listing_cache[key]

    def _resolve_file(self, path: str) -> _HFShard:
        parts, filename = self._split_path(path)
        if filename is None:
            raise IsADirectoryError(f"hf:// path is a directory: {path}")
        shards = self._list_shards(parts)
        shard = shards.get(filename)
        if shard is None:
            raise FileNotFoundError(f"No such hf:// file: {path}")
        return shard

    # ------------------------------------------------------------------
    # StorageBackend protocol
    # ------------------------------------------------------------------

    def exists(self, path: str) -> bool:
        try:
            parts, filename = self._split_path(path)
        except ValueError:
            return False
        try:
            shards = self._list_shards(parts)
        except FileNotFoundError:
            return False
        if filename is None:
            return True
        return filename in shards

    def listdir(self, path: str) -> list[str]:
        parts, filename = self._split_path(path)
        if filename is not None:
            raise NotADirectoryError(f"Not a directory: {path}")
        return sorted(self._list_shards(parts).keys())

    def walk(self, path: str) -> Iterator[tuple[str, int]]:
        """Yield ``(filename, size)`` for every shard under ``path``.

        The ``hf://`` virtual directory is flat — there are no nested
        subdirectories — so the walk emits the same set of files as
        :meth:`listdir`, paired with the sizes already cached from the
        ``/parquet`` listing (no extra HEAD round-trip).
        """
        try:
            parts, filename = self._split_path(path)
        except ValueError:
            return
        if filename is not None:
            return
        try:
            shards = self._list_shards(parts)
        except FileNotFoundError:
            return
        for name in sorted(shards):
            yield name, shards[name].size

    def stat(self, path: str) -> Mapping[str, int | float]:
        shard = self._resolve_file(path)
        # The /parquet endpoint doesn't expose a useful mtime; consumers in
        # zephon only care about ``size``.
        return {"size": shard.size, "mtime": 0.0}

    def read_range(
        self,
        path: str,
        start: int,
        *,
        end: int | None = None,
        length: int | None = None,
    ) -> bytes:
        if start < 0:
            raise ValueError("start must be non-negative")
        if end is not None and length is not None:
            raise ValueError("Specify at most one of end or length")

        shard = self._resolve_file(path)
        if length is not None:
            last = start + length - 1
        elif end is not None:
            last = end - 1
        else:
            last = shard.size - 1 if shard.size > 0 else None

        if last is None or last < start:
            return b""

        headers = self._auth_headers()
        headers["Range"] = f"bytes={start}-{last}"
        # Defeat transparent CDN gzip: if the response body comes back
        # gzip-encoded, the byte offsets of a "range" read no longer map
        # to file offsets, which silently corrupts parquet footer parses.
        headers["Accept-Encoding"] = "identity"

        resp = self._get_session().get(
            shard.url,
            headers=headers,
            timeout=_HTTP_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
        if resp.status_code == 206:
            return resp.content
        if resp.status_code == 200:
            # Server ignored ``Range`` and returned the whole object (some
            # CDNs do this for tiny files or after a redirect). Slice the
            # requested window ourselves so callers always get the bytes
            # they asked for.
            body = resp.content
            return bytes(body[start : last + 1])
        self._raise_for_http_error(resp, path)
        return b""  # unreachable: _raise_for_http_error always raises

    def download(self, src: str, dst: str, timeout: float | None = None) -> None:
        # ``timeout`` is accepted for StorageBackend protocol compatibility
        # but unused: huggingface_hub's ``http_get`` doesn't expose a
        # per-call timeout knob and reads ``HF_HUB_DOWNLOAD_TIMEOUT``
        # internally. Matches the obstore-backed S3/GCS pattern, where the
        # timeout also lives at the store/config level rather than per call.
        del timeout

        shard = self._resolve_file(src)

        try:
            from huggingface_hub.file_download import http_get
            from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for hf:// downloads. "
                "Install with: pip install zephon[hf]"
            ) from exc

        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

        expected_size = shard.size if shard.size > 0 else None
        incomplete = dst + ".incomplete"

        resume_size = 0
        if os.path.exists(incomplete):
            resume_size = os.path.getsize(incomplete)
            # If a previous attempt already finished into ``.incomplete``
            # but the rename was interrupted, just promote it.
            if expected_size is not None and resume_size >= expected_size:
                os.replace(incomplete, dst)
                return

        headers = self._auth_headers() or None
        try:
            with open(incomplete, "ab") as fh:
                http_get(
                    url=shard.url,
                    temp_file=fh,
                    resume_size=resume_size,
                    expected_size=expected_size,
                    headers=headers,
                )
        except EntryNotFoundError as exc:
            self._unlink_quiet(incomplete)
            raise FileNotFoundError(self._not_found_message(src)) from exc
        except HfHubHTTPError as exc:
            resp = getattr(exc, "response", None)
            status = getattr(resp, "status_code", None)
            if status == 404:
                self._unlink_quiet(incomplete)
                raise FileNotFoundError(self._not_found_message(src)) from exc
            if status in (401, 403):
                self._unlink_quiet(incomplete)
                raise PermissionError(
                    f"hf:// access denied: {src}; "
                    "set HF_TOKEN or run `huggingface-cli login`."
                ) from exc
            # Transient HTTP failure (5xx etc.): keep ``.incomplete`` so a
            # caller-driven retry can resume from the current offset.
            raise

        os.replace(incomplete, dst)

    @staticmethod
    def _unlink_quiet(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass

    def glob(self, pattern: str) -> list[str]:
        # Wildcards over hf:// URIs aren't well-defined (configs and splits
        # are enumerated by the Datasets Server, not by path globbing).
        if "*" in pattern or "?" in pattern:
            raise NotImplementedError("Glob patterns are not supported for hf:// URIs")
        return [pattern] if self.exists(pattern) else []

    def put(self, path: str, data: bytes) -> None:
        raise NotImplementedError("hf:// is read-only")

    def delete(self, path: str) -> None:
        raise NotImplementedError("hf:// is read-only")

    def mkdir(self, path: str, parents: bool = False, exist_ok: bool = False) -> None:
        raise NotImplementedError("hf:// is read-only")


__all__ = ["HFBackend"]
