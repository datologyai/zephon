"""HuggingFace dataset storage backend.

Serves an ``hf://`` URI (grammar in :mod:`._hf_uri`) as a flat virtual directory
holding the files of one split:

- :meth:`HFBackend.canonical_root` resolves the URI once (:mod:`._hf_resolve`):
  the split's uploaded files when Zephon reads their format, else HuggingFace's
  Parquet conversion. It returns a frozen URI naming the commit, the source and
  the config, so every later process lists the same files.
- ``listdir`` / ``walk`` list a frozen URI. Names are repo paths with ``/``
  percent-encoded, so the directory is flat and names never collide, even
  across HF's ``train-part{k}`` conversion directories.
- ``stat`` / ``read_range`` / ``download`` turn a frozen root plus a name
  straight into a ``resolve/<commit>/<path>`` URL; they need no listing.

The backend is registered for the ``hf://`` scheme in
:class:`zephon._internal.io.storage.router.RouterStorageBackend`, so the format
handlers read HF datasets without HF-specific code.

Auth: bearer token from ``huggingface_hub.HfFolder.get_token()`` if available,
otherwise the ``HF_TOKEN`` environment variable.
"""

from __future__ import annotations

import logging
import os
import posixpath
import threading
import urllib.parse
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Mapping

from zephon._internal.io.storage._hf_resolve import list_frozen, resolve
from zephon._internal.io.storage._hf_uri import HFUriParts, parse_hf_uri
from zephon._internal.io.storage._utils import OpenViaDownloadMixin
from zephon._internal.io.suffixes import format_of

logger = logging.getLogger(__name__)

_HUB_URL = "https://huggingface.co"
_DATASETS_SERVER_PARQUET_URL = "https://datasets-server.huggingface.co/parquet"
_HTTP_TIMEOUT_SECONDS = 120.0

# Retry transient errors on the Hub and the Datasets Server. The /parquet
# endpoint returns 503 while a dataset is being converted, and the CDN
# occasionally returns 429 under high concurrency.
_RETRY_TOTAL = 4
_RETRY_BACKOFF_FACTOR = 0.5
_RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)

# A partial or still-pending conversion covers only part of the split, so it is
# refused unless the user opts in for a diagnostic run.
_ALLOW_PARTIAL_ENV = "ZEPHON_HF_ALLOW_PARTIAL"

_INCOMPLETE_SUFFIX = ".incomplete"
# The shard cache stages a download at ``<name>.tmp`` and ``download`` writes
# to ``<dst>.incomplete``; both suffixes must fit in one 255-byte file name.
_MAX_NAME_BYTES = 255 - len(".tmp") - len(_INCOMPLETE_SUFFIX)

# Process-wide, shared by every HFBackend: ``Dataset.from_path``, the catalog
# signature and the catalog build each construct their own backend.
_memo_lock = threading.Lock()
_frozen_roots: dict[tuple[str, str | None, bool], str] = {}
_listings: dict[str, dict[str, int]] = {}
_sizes: dict[str, int] = {}
# One lock per in-flight resolution or listing, so concurrent misses on the
# same key wait for one result instead of repeating the Hub requests.
_key_locks: dict[tuple[str, ...], threading.Lock] = {}


def _key_lock(key: tuple[str | bool | None, ...]) -> threading.Lock:
    with _memo_lock:
        return _key_locks.setdefault(tuple(str(part) for part in key), threading.Lock())


# The Hub's paths-info endpoint accepts at most 100 paths per request.
_PATHS_INFO_LIMIT = 100

_TOKEN_UNSET: Any = object()


class HFBackend(OpenViaDownloadMixin):
    """Read-only storage backend for HuggingFace dataset splits.

    All IO is just-in-time: nothing is pre-downloaded. Resolution and listings
    are memoized per process; file access needs only the frozen URI and a name.
    """

    def __init__(self) -> None:
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
                    # POST is only used for the idempotent paths-info lookup.
                    allowed_methods=frozenset({"GET", "HEAD", "POST"}),
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
    def _raise_for_http_error(resp: Any, what: str) -> None:
        """Map HTTP error responses to protocol-conforming exceptions."""
        status = resp.status_code
        if status == 404:
            raise FileNotFoundError(f"hf:// object not found: {what}")
        if status in (401, 403):
            raise PermissionError(
                f"hf:// access denied: {what}; "
                "set HF_TOKEN or run `huggingface-cli login`."
            )
        resp.raise_for_status()

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

    def _get(self, url: str, what: str, **kwargs: Any) -> Any:
        resp = self._get_session().get(
            url, headers=self._auth_headers(), timeout=_HTTP_TIMEOUT_SECONDS, **kwargs
        )
        self._raise_for_http_error(resp, what)
        return resp

    # ------------------------------------------------------------------
    # HubClient (see _hf_resolve)
    # ------------------------------------------------------------------

    def commit_for(self, repo_id: str, revision: str) -> str:
        """Return the 40-hex commit ``revision`` names in ``repo_id``."""
        quoted = urllib.parse.quote(revision, safe="")
        resp = self._get(
            f"{_HUB_URL}/api/datasets/{repo_id}/revision/{quoted}",
            f"hf://{repo_id}@{revision}",
            params={"expand[]": "sha"},
        )
        return str(resp.json()["sha"])

    def conversion_commit(self, repo_id: str) -> str | None:
        """Return the head commit of ``refs/convert/parquet``, if it exists."""
        resp = self._get(f"{_HUB_URL}/api/datasets/{repo_id}/refs", f"hf://{repo_id}")
        for ref in resp.json().get("converts") or []:
            if ref.get("name") == "parquet":
                return str(ref["targetCommit"])
        return None

    def parquet_export(
        self, repo_id: str, config: str | None
    ) -> tuple[Mapping[str, Any], str | None]:
        """Return the Datasets Server ``/parquet`` payload and its ``X-Revision``."""
        params = {"dataset": repo_id}
        if config is not None:
            params["config"] = config
        resp = self._get(
            _DATASETS_SERVER_PARQUET_URL,
            f"parquet conversion of hf://{repo_id}",
            params=params,
        )
        return resp.json(), resp.headers.get("X-Revision")

    def _tree(
        self, repo_id: str, commit: str, path: str, *, recursive: bool
    ) -> Iterator[Mapping[str, Any]]:
        """Yield the tree entries under ``path``, following the Hub's pagination."""
        quoted = f"/{urllib.parse.quote(path)}" if path else ""
        url: str | None = f"{_HUB_URL}/api/datasets/{repo_id}/tree/{commit}{quoted}"
        params = {"recursive": "true", "expand": "false"} if recursive else None
        while url is not None:
            resp = self._get(url, f"hf://{repo_id}@{commit}/{path}", params=params)
            yield from resp.json()
            url = resp.links.get("next", {}).get("url")
            params = None  # the next-page URL carries the query

    def list_tree(
        self, repo_id: str, commit: str, path: str
    ) -> tuple[dict[str, int], list[str]]:
        """Return ``({file name: size}, [subdirectory names])`` directly under ``path``."""
        files: dict[str, int] = {}
        directories: list[str] = []
        for entry in self._tree(repo_id, commit, path, recursive=False):
            name = str(entry["path"]).rsplit("/", 1)[-1]
            if entry.get("type") == "file":
                files[name] = int(entry["size"])
            elif entry.get("type") == "directory":
                directories.append(name)
        return files, directories

    def file_sizes(
        self, repo_id: str, commit: str, paths: Sequence[str]
    ) -> dict[str, int]:
        """Return ``{repo path: size}`` for ``paths`` at ``commit``.

        Up to 100 paths take one paths-info request, so a small split never
        lists the directory it shares with large ones. Larger selections list
        their common directory recursively (1,000 entries per page) and stop as
        soon as every size is known.
        """
        if not paths:
            return {}
        wanted = set(paths)
        sizes: dict[str, int] = {}
        if len(wanted) <= _PATHS_INFO_LIMIT:
            resp = self._get_session().post(
                f"{_HUB_URL}/api/datasets/{repo_id}/paths-info/{commit}",
                data={"paths": sorted(wanted)},
                headers=self._auth_headers(),
                timeout=_HTTP_TIMEOUT_SECONDS,
            )
            self._raise_for_http_error(resp, f"hf://{repo_id}@{commit}")
            entries: Iterable[Mapping[str, Any]] = resp.json()
        else:
            root = posixpath.commonpath([posixpath.dirname(path) for path in paths])
            entries = self._tree(repo_id, commit, root, recursive=True)
        for entry in entries:
            if entry.get("type") == "file" and entry["path"] in wanted:
                sizes[str(entry["path"])] = int(entry["size"])
                if len(sizes) == len(wanted):
                    break  # later tree pages cannot add anything
        missing = [path for path in paths if path not in sizes]
        if missing:
            raise FileNotFoundError(
                f"hf://{repo_id}@{commit} has no {missing[:3]} "
                f"({len(missing)} of {len(paths)} files missing)"
            )
        return {path: sizes[path] for path in paths}

    # ------------------------------------------------------------------
    # Resolution, listings and names
    # ------------------------------------------------------------------

    def canonical_root(self, path: str, fmt: str | None = None) -> str:
        """Resolve ``path`` once and return the frozen URI that pins it.

        ``fmt`` restricts the choice to a source in that format. A frozen URI
        is returned as is, without any request.
        """
        parts = parse_hf_uri(path)
        if parts.is_frozen:
            return parts.uri()
        # The partial opt-in changes the answer, so it is part of the key.
        allow_partial = bool(os.environ.get(_ALLOW_PARTIAL_ENV))
        key = (parts.uri(), fmt, allow_partial)  # normalized: "@main", trailing "/"
        with _memo_lock:
            frozen = _frozen_roots.get(key)
        if frozen is not None:
            return frozen

        with _key_lock(("root", *key)):
            with _memo_lock:
                frozen = _frozen_roots.get(key)
            if frozen is not None:
                return frozen
            resolution = resolve(self, parts, fmt, allow_partial=allow_partial)
            frozen = resolution.parts.uri()
            listing = self._names(resolution.files)
            with _memo_lock:
                _listings.setdefault(frozen, listing)
                _frozen_roots[key] = frozen
        logger.info("Resolved %s to %s (%s)", path, frozen, resolution.summary)
        return frozen

    def _listing(self, root: str) -> dict[str, int]:
        """Return ``{name: size}`` for the split ``root`` names."""
        frozen = self.canonical_root(root)
        with _memo_lock:
            listing = _listings.get(frozen)
        if listing is not None:
            return listing
        with _key_lock(("listing", frozen)):
            with _memo_lock:
                listing = _listings.get(frozen)
            if listing is None:
                listing = self._names(list_frozen(self, parse_hf_uri(frozen)))
                with _memo_lock:
                    _listings[frozen] = listing
        return listing

    @staticmethod
    def _names(files: Mapping[str, int]) -> dict[str, int]:
        """Map repo paths to flat, unique virtual-directory names."""
        names: dict[str, int] = {}
        for path, size in files.items():
            name = urllib.parse.quote(path, safe="")
            if len(name.encode("utf-8")) > _MAX_NAME_BYTES:
                raise ValueError(
                    f"hf:// file {path!r} is too long to cache: its name {name!r} "
                    f"exceeds {_MAX_NAME_BYTES} bytes"
                )
            names[name] = size
        return names

    @staticmethod
    def _split_path(path: str) -> tuple[HFUriParts, str | None]:
        """Return ``(dataset parts, name)``; ``name`` is set only under a frozen root."""
        body = path.rstrip("/")
        parent, _, name = body.rpartition("/")
        try:
            root = parse_hf_uri(parent)
        except ValueError:
            root = None
        if root is not None and root.is_frozen:
            return root, name
        return parse_hf_uri(body), None

    def _resolve_file(self, path: str) -> tuple[HFUriParts, str, str]:
        """Return ``(frozen root, name, URL)`` for a file ``path``."""
        parts, name = self._split_path(path)
        if name is None:
            raise IsADirectoryError(f"hf:// path is a directory: {path}")
        if format_of(name) is None:
            # Only data files are ever listed; a repo's own index.json and the
            # like must not be mistaken for Zephon metadata.
            raise FileNotFoundError(f"No such hf:// file: {path}")
        quoted = urllib.parse.quote(urllib.parse.unquote(name))
        url = f"{_HUB_URL}/datasets/{parts.repo_id}/resolve/{parts.revision}/{quoted}"
        return parts, name, url

    def _size(self, parts: HFUriParts, name: str, url: str) -> int | None:
        """Return the file's size, or ``None`` if it does not exist."""
        with _memo_lock:
            listing = _listings.get(parts.uri())
            if listing is not None:
                return listing.get(name)
            cached = _sizes.get(url)
        if cached is not None:
            return cached

        session = self._get_session()
        headers = self._auth_headers()
        resp = session.head(
            url, headers=headers, timeout=_HTTP_TIMEOUT_SECONDS, allow_redirects=False
        )
        linked = resp.headers.get("X-Linked-Size")
        if linked is None and resp.is_redirect:
            resp = session.head(
                url,
                headers=headers,
                timeout=_HTTP_TIMEOUT_SECONDS,
                allow_redirects=True,
            )
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            self._raise_for_http_error(resp, url)
        size = int(linked if linked is not None else resp.headers["Content-Length"])
        with _memo_lock:
            _sizes[url] = size
        return size

    # ------------------------------------------------------------------
    # StorageBackend protocol
    # ------------------------------------------------------------------

    def exists(self, path: str) -> bool:
        try:
            parts, name, url = self._resolve_file(path)
        except IsADirectoryError:
            try:
                self._listing(path)
            except FileNotFoundError:
                return False
            return True
        except (ValueError, FileNotFoundError):
            return False
        return self._size(parts, name, url) is not None

    def listdir(self, path: str) -> list[str]:
        _, name = self._split_path(path)
        if name is not None:
            raise NotADirectoryError(f"Not a directory: {path}")
        return sorted(self._listing(path))

    def walk(self, path: str) -> Iterator[tuple[str, int]]:
        """Yield ``(name, size)`` for every file of the split ``path`` names.

        The virtual directory is flat, so this is :meth:`listdir` with sizes.
        """
        try:
            _, name = self._split_path(path)
        except ValueError:
            return
        if name is not None:
            return
        try:
            listing = self._listing(path)
        except FileNotFoundError:
            return
        for entry in sorted(listing):
            yield entry, listing[entry]

    def stat(self, path: str) -> Mapping[str, int | float]:
        parts, name, url = self._resolve_file(path)
        size = self._size(parts, name, url)
        if size is None:
            raise FileNotFoundError(f"No such hf:// file: {path}")
        # HF exposes no useful mtime; Zephon only consumes ``size``.
        return {"size": size, "mtime": 0.0}

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

        parts, name, url = self._resolve_file(path)
        if length is not None:
            last: int | None = start + length - 1
        elif end is not None:
            last = end - 1
        else:
            size = self._size(parts, name, url)
            if size is None:
                raise FileNotFoundError(f"No such hf:// file: {path}")
            last = size - 1 if size > 0 else None

        if last is None or last < start:
            return b""

        headers = self._auth_headers()
        headers["Range"] = f"bytes={start}-{last}"
        # Defeat transparent CDN gzip: if the response body comes back
        # gzip-encoded, the byte offsets of a "range" read no longer map
        # to file offsets, which silently corrupts parquet footer parses.
        headers["Accept-Encoding"] = "identity"

        resp = self._get_session().get(
            url,
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

        parts, name, url = self._resolve_file(src)
        expected_size = self._size(parts, name, url)
        if expected_size is None:
            raise FileNotFoundError(f"No such hf:// file: {src}")

        try:
            from huggingface_hub.file_download import http_get
            from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for hf:// downloads. "
                "Install with: pip install zephon[hf]"
            ) from exc

        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

        incomplete = dst + _INCOMPLETE_SUFFIX
        resume_size = 0
        if os.path.exists(incomplete):
            resume_size = os.path.getsize(incomplete)
            # If a previous attempt already finished into ``.incomplete``
            # but the rename was interrupted, just promote it.
            if expected_size > 0 and resume_size >= expected_size:
                os.replace(incomplete, dst)
                return

        headers = self._auth_headers() or None
        try:
            with open(incomplete, "ab") as fh:
                http_get(
                    url=url,
                    temp_file=fh,
                    resume_size=resume_size,
                    expected_size=expected_size or None,
                    headers=headers,
                )
        except EntryNotFoundError as exc:
            self._unlink_quiet(incomplete)
            raise FileNotFoundError(f"No such hf:// file: {src}") from exc
        except HfHubHTTPError as exc:
            resp = getattr(exc, "response", None)
            status = getattr(resp, "status_code", None)
            if status == 404:
                self._unlink_quiet(incomplete)
                raise FileNotFoundError(f"No such hf:// file: {src}") from exc
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
        # are resolved by the Hub, not by path globbing).
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
